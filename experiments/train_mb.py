import f110_gym
import f110_orl_dataset
import gymnasium as gym

from f110_agents.agent import Agent

from torch.utils.data import DataLoader, Subset
from ope_methods.model_based import F110ModelBased, build_dynamics_model
from ope_methods.dataset import F110Dataset, F110DatasetSequence, random_split_indices, model_split_indices
from ope_methods.model_based import DeltaDynamicsModel, SimpleDynamicsModel, ProbDynamicsModel, ProbsDeltaDynamicsModel, AutoregressiveModel
import ope_methods
from functools import partial
import numpy as np
import torch
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm
import argparse


def get_infinite_iterator(dataloader):
    while True:
        for data in dataloader:
            yield data

parser = argparse.ArgumentParser(description='Train model based approaches')


dynamics_models = (
    "AutoregressiveDeltaModel", "SimpleDynamicsModel", "DeltaDynamicsModel", 
    "ProbDynamicsModel", "ProbsDeltaDynamicsModel", "AutoregressiveModel", 
    "EnsembleSDModel", "EnsemblePDDModel", "EnsembleARModel", "EnsembleARDModel"
)

parser.add_argument('--seed', type=int, default=0, help="seed")
parser.add_argument('--eval_interval', type=int, default=5_000, help="eval interval")
parser.add_argument('--update_steps', type=int, default=100_000, help='Number of update steps')
parser.add_argument('--dynamics_model', type=str, default="DeltaDynamicsModel", choices=dynamics_models, help="The possible dynamics models.")
parser.add_argument('--train', action='store_true', help="Enable training.")
parser.add_argument('--save_model', action='store_true', help="Save the model every eval interval.")
parser.add_argument('--skip_eval', action='store_true', help="Skip evaluation step (after saving model).")
parser.add_argument('--model_checkpoint', type=str, default=None, help="Load a model checkpoint.")
args = parser.parse_args()

def evaluate_model(loader, model):
    """
    Evaluates the model on a given DataLoader (test or validation) and returns the average MSE and its standard deviation.

    :param loader: DataLoader for test or validation set.
    :param model: The model to be evaluated.
    :return: Tuple of (average_mse, std_dev_mse)
    """
    mse_values = []

    for batch in loader:
        mse = model.evaluate_ss(batch[0], batch[2], batch[3]) # for not autoregressive models
        mse_values.extend([mse] * len(batch[0]))

    average_mse = sum(mse_values) / len(mse_values)
    std_dev_mse = (sum([(x - average_mse) ** 2 for x in mse_values]) / len(mse_values)) ** 0.5

    return average_mse, std_dev_mse

def evaluate_model_rollouts(loader, model, F110Env, dataset, rollouts=1, horizon=25):
    all_mse = []
    i = 0
    for batch in loader:
        #if i % 2 == 0:
        #    continue # just speed it up a little bit for testing
        #i += 1
        for model_name in np.unique(batch[7]):               
            indices = np.where([str(name)== str(model_name) for name in batch[7] ])[0]
            states = batch[0][indices]
            actor = Agent().load(name=model_name, no_print=True)
            get_target_actions = partial(F110Env.get_target_actions, actor=actor, 
                                        fn_unnormalize_states=dataset.unnormalize_states)
            for i in range(rollouts):
                mse = model.estimate_mse_pose(states, get_target_actions, horizon=horizon)
                all_mse.append(mse)
    return all_mse

import matplotlib.pyplot as plt
def plot_validation_trajectories(behavior_dataset, model, F110Env, test_indices, num_plot, save_path=None, file_name=None, horizon=50):

    finished_sub = behavior_dataset.finished[np.sort(test_indices)]
    states_sub = behavior_dataset.states[np.sort(test_indices)]
    model_names_sub = behavior_dataset.model_names[np.sort(test_indices)]
    # organize the data into trajectories
    starts = np.where(np.roll(finished_sub,1)==1)[0]
    ends = np.where(finished_sub==1)[0]
    # pick num_plot random starts and the corresponding ends
    indices = np.random.choice(len(starts), num_plot)
    starts = starts[indices]
    ends = ends[indices]
    model_names = model_names_sub[indices]
    
    for start, end, model_name in zip(starts, ends,model_names):
        actor = Agent().load(name=model_name, no_print=True)
        get_target_actions = partial(F110Env.get_target_actions, actor=actor, 
                                    fn_unnormalize_states=behavior_dataset.unnormalize_states)
        states = states_sub[start:end+1]
        # add batch dimension
        states = states.unsqueeze(0)
        if horizon is None:
            horizon = states.shape[1]
        actions = torch.zeros((1, 1, 2))
        rollout_states, actions = model.rollout(states, actions, get_target_actions, horizon=horizon)
        plt.plot(rollout_states[0,:horizon,0], rollout_states[0,:horizon,1], color="red")
        plt.plot(states[0,:horizon,0], states[0,:horizon,1], color="blue")
    if save_path is not None:
        # show the legend
        plt.legend(["rollout", "ground truth"])
        # x andy y labels
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("Rollout vs. Ground Truth trajectories")
        plt.savefig(os.path.join(save_path, file_name))
        plt.clf()
    # plt.show()



def compute_mse_rollouts(behavior_dataset, model, F110Env, test_indices, rollouts=1):

    finished_sub = behavior_dataset.finished[np.sort(test_indices)]
    states_sub = behavior_dataset.states[np.sort(test_indices)]
    model_names_sub = behavior_dataset.model_names[np.sort(test_indices)]
    # organize the data into trajectories
    starts = np.where(np.roll(finished_sub,1)==1)[0]
    ends = np.where(finished_sub==1)[0]

    max_length = max(ends +1 -starts)

    trajectories = torch.zeros((len(starts), max_length, states_sub.shape[1]))
    truncation = torch.ones((len(starts))) * max_length
    model_name_trajectories = []
    truncations = []
    for i, (start, end) in enumerate(zip(starts, ends)):
        trajectories[i,:end + 1 - start,:] = states_sub[start:end+1, :]
        truncations.append(int(end - start))
        model_name_trajectories.append(model_names_sub[start])

    #for i in range(trajectories.shape[0]):
    #    plt.plot(trajectories[i,:int(truncation[i]),0], trajectories[i,:int(truncation[i]),1])
    
    model_name_trajectories = np.array(model_name_trajectories)
    truncations = np.array(truncations)
    all_mse = torch.zeros((max_length))
    for model_name in np.unique(model_name_trajectories):               
        indices = np.where([str(name)== str(model_name) for name in  model_name_trajectories])[0]
        states_picked = trajectories[indices]
        relevant_truncations = truncations[indices]
        actor = Agent().load(name=model_name, no_print=True)
        get_target_actions = partial(F110Env.get_target_actions, actor=actor, 
                                    fn_unnormalize_states=behavior_dataset.unnormalize_states)
        rollout_mse = torch.zeros((max_length))
        for i in range(rollouts):
            actions = torch.zeros((states_picked.shape[0], 1, 2))
            rollout_states, actions = model.rollout(states_picked, actions, get_target_actions, horizon=max_length)
            #for i, rollout_state in enumerate(rollout_states):
            #    plt.plot(rollout_state[:int(relevant_truncations[i]),0], rollout_state[:int(relevant_truncations[i]),1], color="red")
            #    plt.plot(states_picked[i,:int(relevant_truncations[i]),0], states_picked[i,:int(relevant_truncations[i]),1], color="blue")
            #plt.show()
            new_states = torch.zeros_like(rollout_states)
            for i in range(len(rollout_states)):
                new_states[i,:int(relevant_truncations[i])] = rollout_states[i,:int(relevant_truncations[i]),:]
            mse = rollout_states - states_picked
            # sum over the last dimension
            mse = torch.sum(mse**2, dim = 0)
            mse = torch.sum(mse, dim = -1)
            rollout_mse += mse
        rollout_mse /= rollouts
        all_mse += rollout_mse
    all_mse /= len(np.unique(model_name_trajectories))
    #plt.plot(all_mse)
    #plt.show()
    return all_mse


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_path = ope_methods.dataset.create_save_dir(
        experiment_directory = "runs_mb",
        algo=args.dynamics_model,
        reward_name="reward_progress",
        dataset="f110-real-stoch-v2",
        target_policy="off-policy",
        seed = args.seed,
    )
    print("Logging to :", save_path)
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
        include_timesteps_in_obs = False,
        set_terminals=True,
        delta_factor=1.0,
        reward_config=None,#"reward_progress.json",
        include_pose_time_diff=False,
        include_action_pose_time_diff = False,
        include_time_obs = False,
        include_progress=False,
        set_previous_step_terminals=0,
        use_compute_termination=True,
        remove_cons_terminals=True,
        **dict(name="f110-real-stoch-v2",
            config = dict(map="Infsaal3", num_agents=1,
            params=dict(vmin=0.0, vmax=2.0)),
            render_mode="human")
    )
    
    behavior_dataset = F110Dataset(
        F110Env,
        normalize_states=True,
        normalize_rewards=False,
        train_only = True,
       # only_agents = [args.agent],
    )
    print("Loaded training dataset.")
    
    eval_dataset = F110Dataset(
        F110Env,
        normalize_states=True,
        normalize_rewards=False,
        only_agents = F110Env.eval_agents,
        state_mean=behavior_dataset.state_mean,
        state_std=behavior_dataset.state_std,
        reward_mean=behavior_dataset.reward_mean,
        reward_std=behavior_dataset.reward_std,
    )
    print("Loaded evaluation dataset.")
    
    val_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False)
    train_loader = DataLoader(behavior_dataset, batch_size=256, shuffle=True)
    
    inf_dataloader = get_infinite_iterator(train_loader)
    data_iter = iter(inf_dataloader)


    
    min_states = behavior_dataset.states.min(axis=0)[0]
    max_states = behavior_dataset.states.max(axis=0)[0]
    min_states[:2] = min_states[:2] - 2.0
    max_states[:2] = max_states[:2] + 2.0
    dynamics_model = build_dynamics_model(args.dynamics_model,
                                          min_state=min_states, 
                                          max_state=max_states)
    
    ### Define the Model ### 
    model = F110ModelBased(F110Env, behavior_dataset.states.shape[1],
                behavior_dataset.actions.shape[1],
                dynamics_model = dynamics_model,
                hidden_size = [256,256,256,256],
                dt=1/20,
                min_state=min_states,
                max_state=max_states,
                fn_normalize=behavior_dataset.normalize_states,
                fn_unnormalize=behavior_dataset.unnormalize_states,
                use_reward_model=False,
                use_done_model=False,
                obs_keys=behavior_dataset.obs_keys,
                learning_rate=1e-3,
                weight_decay=1e-4,
                target_reward="reward_progress.json",
                logger=writer,)

    if args.model_checkpoint is not None:
        model.load(save_path, args.model_checkpoint)
    
    pbar = tqdm(range(args.update_steps), mininterval=5.0)

    for i in pbar:
        if args.train:
            model.set_device("cuda")
            (states, scans, actions, next_states, next_scans, rewards, masks, sequence_model_names,
            log_prob) = next(data_iter)
            assert not np.isnan(states).any()
            assert not np.isnan(actions).any()
            assert not np.isnan(next_states).any()
            assert not np.isnan(rewards).any()
            assert not np.isnan(masks).any()

            loss, loss_reward, loss_done = model.update(states, actions, 
                                                        next_states, rewards,
                                                        masks)
            
            writer.add_scalar(f"train/loss_mb", loss, global_step=i)
            writer.add_scalar(f"train/loss_reward", loss_reward, global_step=i)
            writer.add_scalar(f"train/loss_done", loss_done, global_step=i)

        if i % args.eval_interval == 0:
            num_plot = 10
            model.set_device("cpu")
            #plot_validation_trajectories(behavior_dataset, model, F110Env, val_indices, num_plot, save_path=save_path, file_name=f"rollouts_{i}")

            if args.save_model:
                model.save(save_path, filename=f"model_{args.seed}_{i}.pth")
            if args.skip_eval:
                continue
            
            average_mse_val, std_dev_mse_val = evaluate_model(val_loader, model)
            writer.add_scalar(f"eval/mse_single", average_mse_val, global_step=i)
            writer.add_scalar(f"eval/std_mse_val", std_dev_mse_val, global_step=i)

if __name__ == "__main__":
    main(args)