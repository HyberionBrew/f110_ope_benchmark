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

parser = argparse.ArgumentParser(description='Train and eval FQE models')

parser.add_argument('--seed', type=int, default=0, help="seed")
parser.add_argument('--eval_interval', type=int, default=5_000, help="Evaluation interval.")
parser.add_argument('--update_steps', type=int, default=200_000, help='Total number of update steps')
parser.add_argument('--fqe_model', type=str, default="QFitterL2", choices=["QFitterDD", "QFitterL2"], 
                    help="FQE version to use.")
parser.add_argument('--target_reward', type=str, default="reward_progress.json", help="The target reward, see .bash script for options.")
parser.add_argument('--discount', type=float, default=0.99, help="The Discount factor for the sum of discounted rewards.")
parser.add_argument('--train', action='store_true', help="Enable training.")
parser.add_argument('--save_model', action='store_true', help="Save the model every 10_000 steps.")
parser.add_argument('--agent',type=str, default="StochasticContinousFTGAgent_0.6_2_0.8_0.03_0.1_5.0_0.3_0.5", help="The Agent we want to run FQE for. See .bash script for options.")
parser.add_argument('--lr', type=float, default=1e-4, help="The learning rate.")
parser.add_argument('--tau', type=float, default=0.005, help="Soft update tau parameter")
parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay for the optimizer.")
parser.add_argument('--eval_only', action='store_true', help="Eval only, no training.")
args = parser.parse_args()


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    save_path = ope_methods.dataset.create_save_dir(
        experiment_directory = f"runs_fqe_{args.lr}_{args.tau}_{args.weight_decay}_{args.target_reward}",
        algo=args.fqe_model,
        reward_name=args.target_reward,
        dataset="f110-real-stoch-v2",
        target_policy="on-policy", # this should actually read "off-policy", this only changes the folder name and is inconsequential
        seed = args.seed,
    )
    print("Logging to", save_path)
    # append the args.agent to the save path
    save_path = os.path.join(save_path, args.agent) 
    import datetime
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d-%H-%M-%S")
    file_name = str(args.seed) + "_" + time
    writer = SummaryWriter(log_dir= os.path.join(save_path, file_name))
    
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
        train_only = True,
    )

    print("inital states training:", torch.sum(training_dataset.mask_inital))
    actor = Agent().load(name=args.agent, no_print=True)
    get_target_actions = partial(F110Env.get_target_actions, actor=actor, 
                                fn_unnormalize_states=training_dataset.unnormalize_states)
    # compute the next actions
    next_actions = get_target_actions(training_dataset.states_next)
    training_dataset.next_actions = next_actions
    
    print(args)
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
    initial_eval_actions = get_target_actions(initial_eval_states)

    train_loader = DataLoader(training_dataset, batch_size=256, shuffle=True)

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
                tau= args.tau,
                #critic_lr=1e-6, 
                #weight_decay=1e-7,
                #tau=0.005,
                discount=args.discount, 
                logger=writer)
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
                logger=writer)
    elif args.fqe_model == "QFitterDD":
        model = QFitterDD(training_dataset.states.shape[1], 
                training_dataset.actions.shape[1], 
                hidden_sizes=[256,256,256,256], 
                num_atoms=101, 
                min_reward=min_reward, 
                max_reward=max_reward, 
                critic_lr=args.lr,
                weight_decay=args.weight_decay,
                tau = args.tau,
                discount=args.discount, 
                logger=writer)

        
    import time
    
    i = 0
    pbar = tqdm(total=args.update_steps)
    
    while i < args.update_steps:
        # initalize the datset
        next_actions = get_target_actions(training_dataset.states_next)
        training_dataset.next_actions = next_actions
        train_loader = DataLoader(training_dataset, batch_size=256, shuffle=True)

        for step, (states, scans, actions, next_states, next_scans, rewards, masks, sequence_model_names,
            log_prob, next_actions) in enumerate(train_loader):
            if args.train:
                model.set_device("cuda")
                assert not np.isnan(states).any()
                assert not np.isnan(actions).any()
                assert not np.isnan(next_states).any()
                assert not np.isnan(rewards).any()
                assert not np.isnan(masks).any()
                
                # print(masks.sum())
                loss = model.update(states,actions, 
                                    next_states, 
                                    next_actions,
                                    rewards, 
                                    masks)
                writer.add_scalar(f"train/loss_reward", loss, global_step=i)
                writer.add_scalar(f"train/loss", loss, global_step=i)

                if args.save_model and i%10_000 == 0:
                    model.save(save_path, i)

            if i % args.eval_interval == 0:
                #print(save_path)
                #model.load(save_path, i)
                model.set_device("cuda")
                
                if args.eval_only:
                    model.load(save_path, i=190000)
                    model.set_device("cuda")
                    mean, std = model.estimate_returns(initial_eval_states, initial_eval_actions, plot=True)
                    result_dict = dict()
                    result_dict[args.agent] = {"mean": mean.item(),
                                    "std": std.item()}
                    print(result_dict)
                    save_path = os.path.join(save_path, "results")

                    exit()


                if i % 2_000 == 0 and args.fqe_model == "QFitterDD":
                    mean, std = model.estimate_returns(initial_eval_states, initial_eval_actions, plot=True)
                else:
                    mean, std = model.estimate_returns(initial_eval_states, initial_eval_actions)
                    
                writer.add_scalar(f"eval/mean", mean.item(), global_step=i)
                writer.add_scalar(f"eval/std", std.item(), global_step=i)
                # add the mean to pbar
                pbar.set_postfix({"mean": mean.item(), "std": std.item()})

            i += 1
            pbar.update(1)  # Manually update the tqdm progress bar
        
            # Check if we've reached or exceeded the update steps to break the outer loop as well
            if i >= args.update_steps:
                break
       
if __name__ == "__main__":
    main(args)