import f110_gym
import f110_orl_dataset
import gymnasium as gym

from f110_agents.agent import Agent

from torch.utils.data import DataLoader, Subset
from ope_methods.fqe import QFitterBase, QFitterL2, QFitterLME, QFitterDD
from ope_methods.dataset import F110Dataset, F110DatasetSequence, random_split_indices, model_split_indices
from ope_methods.iw import ImportanceSamplingContinousStart
import ope_methods
from functools import partial
import numpy as np
import torch
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm
import argparse
from create_plots import plot_bars_from_dicts
import json
import matplotlib.pyplot as plt
from scipy.stats import norm


def get_infinite_iterator(dataloader):
    while True:
        for data in dataloader:
            yield data

    

parser = argparse.ArgumentParser(description='Run IW')
iw_choices=[ "step_wis_termination" "phwis_heuristic" "phwis" "step_wis" "simple_is" "simple_step_is" "cobs_wis"]

parser.add_argument('--iw_type', type=str, default="step_wis",choices=iw_choices, help="Type of importance sampling to use.")
parser.add_argument('--target_reward', type=str, default="reward_progress.json", help="The target reward. See the associated bash script for valid values.")
parser.add_argument('--discount', type=float, default=0.99, help="The discount factor to compute the discounted rewards.")
parser.add_argument('--save', action='store_true', help="Save the output as a file.")
parser.add_argument('--plot', action='store_true', help="Create some result plots.")
parser.add_argument('--seed', type=int, default=-1, help="Seed, relevant for DR only.")
parser.add_argument('--ext', type=str, default="zero", help="Extension method, can utilize 'mean' as well to compute PDWIS (Mean).")
parser.add_argument('--dr', action='store_true', help="Use DR method, needs a pre-trained FQE model.")
args = parser.parse_args()


def compute_prob_trajectories(log_probs, finished, model_names, rewards):
    starts = np.where(np.roll(finished, 1) == 1)[0]
    ends = np.where(finished == 1)[0]
    horizon = np.max(ends + 1 -starts)
    prob_trajectories = np.zeros((len(starts), horizon, log_probs.shape[1]))
    terminations_ = np.zeros(len(starts),dtype=int)
    model_names_ = []
    rewards_ = np.zeros((len(starts), horizon))
    for i, (start, end) in enumerate(zip(starts, ends)):
        #print(start,end)
        #print(val_dataset.states[start:end+1].shape)
        prob_trajectories[i, 0:end - start+ 1] = log_probs[start:end+1]
        term = np.where(finished[start:end+1])[0]
        model_names_.append(model_names[start])
        if len(term )== 0:
            term = [horizon+1]
        rewards_[i, 0:end - start+ 1] = rewards[start:end+1]
        terminations_[i] = int(term[0])
    return prob_trajectories, terminations_, model_names_ ,rewards_


def main(args):

    log_prob_type = "action"
    algo_name = f"iw_{log_prob_type}_{args.iw_type}_{args.ext}"
    # add _dr if dr is used
    if args.dr:
        algo_name = algo_name + "_dr"
    save_path = ope_methods.dataset.create_save_dir(
        experiment_directory = f"runs_iw" if not(args.dr) else f"runs_dr",
        algo= algo_name,
        reward_name="reward_progress",
        dataset="f110-real-stoch-v2",
        target_policy="off-policy",
        seed = args.seed,
    )
    print("Logging to: ", save_path)

    import datetime
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d-%H-%M-%S")

    
    F110Env = gym.make("f110-real-stoch-v2",
        encode_cyclic=True,
        flatten_obs=True,
        timesteps_to_include=(0,250),
        use_delta_actions=True,
        set_terminals=True,
        delta_factor=1.0,
        reward_config=args.target_reward,#"reward_progress.json",
        include_pose_time_diff=False,
        include_action_pose_time_diff = False,
        include_time_obs = True,
        include_progress=False,
        set_previous_step_terminals=0,
        use_compute_termination=True,
        remove_cons_terminals=True,
        **dict(name="f110-real-stoch-v2",
            config = dict(map="Infsaal3", num_agents=1,
            params=dict(vmin=0.0, vmax=2.0)),
            render_mode="human")
    )

    ### get the dataset ###
    training_dataset = F110Dataset(
        F110Env,
        normalize_states=True,
        normalize_rewards=False,
        train_only=True,
    )
    print("Loaded training dataset")
    print("inital states training:", torch.sum(training_dataset.mask_inital))

    result_dict = {}
    
    from tqdm import tqdm
    for agent_num, agent in tqdm(enumerate(F110Env.eval_agents)): 
        eval_dataset = F110Dataset(
            F110Env,
            normalize_states=True,
            normalize_rewards=False,
            only_agents = [agent],
            state_mean=training_dataset.state_mean,
            state_std=training_dataset.state_std,
            reward_mean=training_dataset.reward_mean,
            reward_std=training_dataset.reward_std,
        )
        print(f"Loaded eval dataset for agent: {agent}")
        if log_prob_type == "action":
            actor = Agent().load(name=agent, no_print=True)
            get_log_probs = partial(F110Env.get_target_log_probs, 
                                    actor=actor, 
                                    fn_unnormalize_states=training_dataset.unnormalize_states)
            # compute the next actions
            target_log_probs = get_log_probs(training_dataset.states,
                                            training_dataset.actions,
                                            scans = training_dataset.scans)

            behavior_log_probs = training_dataset.log_probs.reshape(training_dataset.log_probs.shape[0], -1)
        

        start_points = training_dataset.unnormalize_states(training_dataset.states[training_dataset.mask_inital])
        start_scans = training_dataset.scans[training_dataset.mask_inital]
        # need to transform the log probs and target probs to trajectories
        finished = training_dataset.finished
        start_points_eval =  training_dataset.unnormalize_states(eval_dataset.states[eval_dataset.mask_inital])
        
        offset = 0.0
        train_rewards = training_dataset.rewards
        get_target_actions=None
        model_fqe=None
        if args.dr:
            model_fqe = QFitterDD(training_dataset.states.shape[1], 
                training_dataset.actions.shape[1], 
                hidden_sizes=[256,256,256,256], 
                num_atoms=101, 
                min_reward=0.0, 
                max_reward=100.0, 
                critic_lr=100, # no training anyways
                weight_decay=1e-5,
                tau = 0.005,
                discount=args.discount, 
                logger=None)


            model_fqe.load(f"runs_fqe_0.0001_0.005_1e-05_{args.target_reward}/QFitterDD/f110-real-stoch-v2/250/on-policy/{args.seed}/{agent}", i=190000)

            actor = Agent().load(name=agent, no_print=True)
            print("Getting starting actions")
            get_target_actions = partial(F110Env.get_target_actions, actor=actor, 
                                fn_unnormalize_states=training_dataset.unnormalize_states)

            
            model_fqe.set_device("cuda")

            batch_size = 1000
            n_batches = len(training_dataset.states) // batch_size
            all_q_values = torch.zeros((len(training_dataset.states)))
            for i in range(n_batches):

                _, _ , q_values = model_fqe.estimate_returns(training_dataset.states[i*batch_size:(i+1)*batch_size].cuda(), 
                                                             training_dataset.actions[i*batch_size:(i+1)*batch_size].cuda(),
                                                             get_q_vals=True)
                all_q_values[i*batch_size:(i+1)*batch_size] = q_values
            q_values = all_q_values.cpu().detach().numpy()

            n_samples = 5

            print("Start Q-Estimation")
            all_sample_actions = []
            for s in range(n_samples):

                all_next_actions = torch.zeros_like(training_dataset.actions)
                for i in range(n_batches):
                    next_actions = get_target_actions(training_dataset.states_next[i*batch_size:(i+1)*batch_size], 
                                                    scans=training_dataset.scans_next[i*batch_size:(i+1)*batch_size])
    
                    all_next_actions[i*batch_size:(i+1)*batch_size] = next_actions
                all_sample_actions.append(all_next_actions)
                
            next_actions = all_sample_actions

            print("Got next actions")
            all_next_q_values = np.zeros((len(training_dataset.states)))
            for i in range(n_batches):
                #print(i)
                next_q_values = sum(
                    [model_fqe.estimate_returns(training_dataset.states_next[i*batch_size:(i+1)*batch_size].cuda(), 
                                                next_action[i*batch_size:(i+1)*batch_size].cuda(), 
                                                get_q_vals=True)[2] for next_action in next_actions]) / n_samples    
                all_next_q_values[i*batch_size:(i+1)*batch_size] = next_q_values
                #print(next_q_values.shape)
            next_q_values = all_next_q_values
            #print(q_values.shape)
            #print(train_rewards.shape)
            train_rewards = train_rewards + args.discount * next_q_values - q_values
            print("Finished Q-Estimation")

        behavior_log_probs, terminations_behavior, behavior_agent_names, rewards = compute_prob_trajectories(behavior_log_probs, finished, training_dataset.model_names,train_rewards)
        target_log_probs, terminatons_target, _ , rewards = compute_prob_trajectories(target_log_probs, finished, ["target"]*len(target_log_probs), train_rewards)

        # clip behavior and target log_probs 
        # this is done as in DOPE
        behavior_log_probs = np.clip(behavior_log_probs, -7, 2)
        target_log_probs = np.clip(target_log_probs, -7, 2)

        behavior_log_probs = np.sum(behavior_log_probs, axis=2)
        target_log_probs = np.sum(target_log_probs, axis=2)

        trajectories, actions, terminations , model_names = F110Env.compute_trajectories(training_dataset.states, training_dataset.actions, training_dataset.finished,training_dataset.finished, training_dataset.model_names)
        min_idx = np.argmin(terminations)

        reward = ImportanceSamplingContinousStart(behavior_log_probs, 
                                        target_log_probs, 
                                        np.array([str(ag) for ag in behavior_agent_names]),
                                        terminations_behavior,
                                        rewards,
                                        start_points.numpy(),
                                        start_points_eval.numpy(),
                                        start_distance=1.0, 
                                        start_prob_method = "l2",
                                        plot=False,
                                        agent_name = agent,
                                        iw_type=args.iw_type,
                                        fill_type=args.ext,
                                        get_actions=get_target_actions,
                                        model=model_fqe,
                                        start_scans=start_scans,
                                        normalize_states=training_dataset.normalize_states,
                                        )
        

        reward = reward + offset
        reward = reward
        print(f"Predicted {agent}: {reward}")
        result_dict[agent] = {"mean": reward, "std": 0.0}
        # break
        
    if args.plot:
        print(result_dict)
        plot_bars_from_dicts([result_dict], ["Rollouts"], f"Mean discounted reward {args.target_reward}",plot=True)
    # add result to the save path
    path_res = "results"
    save_path = os.path.join(save_path, path_res)
    if args.save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, f"{args.target_reward}"), 'w') as file:
            json.dump(result_dict, file)
                

# main
if __name__ == "__main__":
    main(args)