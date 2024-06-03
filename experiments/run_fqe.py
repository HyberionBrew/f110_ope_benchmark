import f110_gym
import f110_orl_dataset
import gymnasium as gym

from f110_agents.agent import Agent

from torch.utils.data import DataLoader, Subset
from ope_methods.fqe import QFitterBase, QFitterL2, QFitterLME, QFitterDD
from ope_methods.dataset import F110Dataset, F110DatasetSequence, random_split_indices, model_split_indices

import ope_methods
from functools import partial
import numpy as np
import torch
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm
import argparse
import json


def get_infinite_iterator(dataloader):
    while True:
        for data in dataloader:
            yield data

parser = argparse.ArgumentParser(description='Train model based approaches')

parser.add_argument('--seed', type=int, default=0, help="seed")
parser.add_argument('--eval_interval', type=int, default=5_000, help="eval interval")
parser.add_argument('--eval_samples', type=int, default=20, help='eval samples')
parser.add_argument('--split', type=str, default="on-policy", help="split")
parser.add_argument('--fqe_model', type=str, default="QFitterL2", help="dynamics model")
parser.add_argument('--target_reward', type=str, default="reward_progress.json", help="target reward")
parser.add_argument('--discount', type=float, default=0.99, help="discount factor")
parser.add_argument('--train', action='store_true', help="train")
parser.add_argument('--skip_eval', action='store_true', help="skip eval")
parser.add_argument('--model_checkpoint', type=str, default=None, help="model checkpoint")
parser.add_argument('--agent',type=str, default="StochasticContinousFTGAgent_0.6_2_0.8_0.03_0.1_5.0_0.3_0.5", help="agent")
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--tau', type=float, default=0.005, help="tau")
parser.add_argument('--weight_decay', type=float, default=0.00001, help="dynamics model")
parser.add_argument('--eval_only', action='store_true', help="eval only")
args = parser.parse_args()


def sanity_check(F110Env, eval_dataset):
    print(eval_dataset.masks.sum())
    model_names = eval_dataset.model_names
    rewards = dict()
    min_reward, max_reward = 0, 0
    for model in np.unique(eval_dataset.model_names):
        model_trajectories = eval_dataset.states[model_names == model]
        model_action_trajectories = eval_dataset.actions[model_names == model]
        ends = np.where(eval_dataset.finished[model_names == model])[0]
        starts = np.where(eval_dataset.mask_inital[model_names == model])[0]
        rewards[model] = 0
        for start, end in zip(starts,ends):
            unnormalized_model_traj = eval_dataset.unnormalize_states(model_trajectories[start:end+1])
            reward = F110Env.compute_reward_trajectories(unnormalized_model_traj.unsqueeze(0), 
                                                         model_action_trajectories[start:end+1].unsqueeze(0), 
                                                         torch.tensor([len(model_trajectories[start:end+1])]).unsqueeze(0), 
                                                         "reward_progress.json")
            discount_factors = args.discount ** np.arange(reward.shape[1])
            # Calculate the sum of discounted rewards along axis 1
            discounted_sums = np.sum(reward * discount_factors, axis=1)[0]
            if min_reward > discounted_sums:
                min_reward = discounted_sums
            if max_reward < discounted_sums:
                max_reward = discounted_sums
            rewards[model] += discounted_sums
        rewards[model] /= len(starts)
    print(rewards)
    return min_reward, max_reward

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(args)
    if len(args.agent) < 6:
        exit()

    save_path = ope_methods.dataset.create_save_dir(
        experiment_directory = f"runs_fqe_{args.lr}_{args.tau}_{args.weight_decay}_{args.target_reward}",
        algo=args.fqe_model,
        reward_name=args.target_reward,
        dataset="f110-real-stoch-v2",
        target_policy=args.split,
        seed = args.seed,
    )
    print(save_path)
    # append the args.agent to the save path
    save_path = os.path.join(save_path, args.agent) 
    import datetime
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d-%H-%M-%S")
    file_name = str(args.seed) + "_" + time
    
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
    # needed for the unnormalization computation
    training_dataset = F110Dataset(
        F110Env,
        normalize_states=True,
        normalize_rewards=False,
        train_only = True,
       # only_agents = [args.agent],
    )

    print("inital states training:", torch.sum(training_dataset.mask_inital))
    actor = Agent().load(name=args.agent, no_print=True)
    get_target_actions = partial(F110Env.get_target_actions, actor=actor, 
                                fn_unnormalize_states=training_dataset.unnormalize_states)
    # compute the next actions
    next_actions = get_target_actions(training_dataset.states_next)
    training_dataset.next_actions = next_actions
    #print(training_dataset.next_actions[:20])
    #print(training_dataset.actions[:20])
    #exit()
    ### done ###

    eval_dataset = F110Dataset(
        F110Env,
        normalize_states=True,
        normalize_rewards=False,
        only_agents = [args.agent],
        state_mean=training_dataset.state_mean,
        state_std=training_dataset.state_std,
        reward_mean=training_dataset.reward_mean,
        reward_std=training_dataset.reward_std,
    )
    print("inital states eval:", torch.sum(eval_dataset.mask_inital))
    initial_eval_states = eval_dataset.states[np.where(eval_dataset.mask_inital)[0]]
    #initial_eval_actions = get_target_actions(initial_eval_states)

    #train_loader = DataLoader(training_dataset, batch_size=256, shuffle=True)
    # inf_dataloader = get_infinite_iterator(train_loader)
    ###### Sanity Checks ######
     #min_reward, max_reward = sanity_check(F110Env, eval_dataset)
    # this is appropriate for all of our rewards tbh, but make variable latter
    min_reward = training_dataset.normalize_rewards(0)
    max_reward = training_dataset.normalize_rewards(100) # max(100, max_reward)
    print(torch.std(training_dataset.rewards))
    print("min and max reward")
    print(min_reward, max_reward)
    # normalize min and max_reward according to mean and std
    min_reward = (min_reward - training_dataset.reward_mean) / training_dataset.reward_std
    max_reward = (max_reward - training_dataset.reward_mean) / training_dataset.reward_std

    # build the model
    if args.fqe_model == "QFitterL2":
        model = QFitterL2(training_dataset.states.shape[1], 
                training_dataset.actions.shape[1], 
                hidden_sizes=[256,256,256,256], # 256,256,256,256
                min_reward = min_reward,
                max_reward = max_reward,
                average_reward = 30.0,
                critic_lr=args.lr, 
                weight_decay=args.weight_decay,
                tau=args.tau,
                #critic_lr=1e-6, 
                #weight_decay=1e-7,
                #tau=0.005,
                discount=args.discount, 
                logger=None)
    if args.fqe_model == "QFitterLME":
        model = QFitterLME(training_dataset.states.shape[1], 
                training_dataset.actions.shape[1], 
                hidden_sizes=[256,256,256,256], 
                min_reward = min_reward,
                max_reward = max_reward,
                average_reward = 30.0,
                critic_lr=args.lr, 
                weight_decay=args.weight_decay,
                tau=args.tau,
                discount=args.discount, 
                logger=None)
    elif args.fqe_model == "QFitterDD":
        model = QFitterDD(training_dataset.states.shape[1], 
                training_dataset.actions.shape[1], 
                hidden_sizes=[256,256,256,256], 
                num_atoms=101, 
                min_reward=min_reward, 
                max_reward=max_reward, 
                critic_lr=args.lr, #1e-5 -> works
                weight_decay=args.weight_decay,
                tau = args.tau,
                discount=args.discount, 
                logger=None)

        
    # pbar = tqdm(range(args.update_steps))#, mininterval=5.0)


    #for i in pbar:
    i = 0
    #pbar = tqdm(total=args.eval_samples)
    model.load(save_path, i=190000)
    model.set_device("cuda")
    means = []
    stds = []
    while i < args.eval_samples:
        # initalize the datset
        actions = get_target_actions(initial_eval_states)

        
        mean, std = model.estimate_returns(initial_eval_states, actions)
        result_dict = dict()
        means.append(mean.item())
        stds.append(std.item())

        #save_path = os.path.join(save_path, "results")
        #pbar.set_postfix({"mean": mean.item(), "std": std.item()})

            #writer.add_scalar(f"train/loss_done", loss_done, global_step=i)
        i += 1
        #pbar.update(1)  # Manually update the tqdm progress bar
    print("Mean: ", np.mean(means))
    print("Std: ", np.mean(stds))
    # add add save path results dict and save
    result_dict[args.agent] = dict()
    result_dict[args.agent]["mean"] = np.mean(means)
    result_dict[args.agent]["std"] = np.mean(stds)
    # create the results folder if it does not exist
    if not os.path.exists(os.path.join(save_path, "results")):
        os.makedirs(os.path.join(save_path, "results"))
    # save with the name of the reward as json
    with open(os.path.join(save_path, "results", args.target_reward), 'w') as f:
        json.dump(result_dict, f, indent=4)

if __name__ == "__main__":
    main(args)