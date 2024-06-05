import f110_gym
import f110_orl_dataset
import gymnasium as gym
import numpy as np

import argparse
import matplotlib.pyplot as plt
import json
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Train model based approaches')

# target reward
parser.add_argument('--target_reward', type=str, default="reward_progress.json", help="target reward")
parser.add_argument('--discount', type=float, default=0.99, help="discount factor")
parser.add_argument('--output_folder', type=str, default="runs3", help="where to save the ground truth rewards")
parser.add_argument('--zarr_path', type=str, default=None, help="path to the zarr file if not using default")
args = parser.parse_args()
from create_plots import plot_bars_from_dicts

def main(args):
    # load the dataset:
    F110Env = gym.make("f110-real-stoch-v2",
        encode_cyclic=True,
        flatten_obs=True,
        timesteps_to_include=(0,250),
        use_delta_actions=True,
        include_timesteps_in_obs = False,
        set_terminals=True,
        delta_factor=1.0,
        reward_config=None,
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
    print(F110Env.eval_agents)
    #if args.zarr_path is not None:
    #    dataset = F110Env.get_dataset(zarr_path = args.zarr_path,eval_only=True)
    #else:
    dataset = F110Env.get_dataset(eval_only=True)

    ground_truth_rewards = {}
    trajectories, action_trajectories, terminations, model_names = F110Env.compute_trajectories(dataset["observations"],
                                                                                                dataset["actions"],
                                                                                                dataset["terminals"],
                                                                                                dataset["timeouts"], 
                                                                                                dataset["model_name"])

    #(829, 250, 9)
    print(trajectories.shape)
    #exit()
    print(len(np.unique(dataset["model_name"])))
    for model in np.unique(dataset["model_name"]): #tqdm(F110Env.eval_agents):
        model_trajectories = trajectories[model_names == model]
        print(model_trajectories.shape)
        if model_trajectories.shape[0] < 5:
            continue
        model_action_trajectories = action_trajectories[model_names == model]
        model_terminations = terminations[model_names == model]
        #print(len(model_trajectories))
        #print(model
        reward = F110Env.compute_reward_trajectories(model_trajectories, model_action_trajectories, model_terminations, args.target_reward)
        discount_factors = args.discount ** np.arange(trajectories.shape[1])
        # Calculate the sum of discounted rewards along axis 1
        discounted_sums = np.sum(reward * discount_factors, axis=1)
        # print(f"Model: {model}, Reward: {reward}")
        ground_truth_rewards[model] = {
            "mean": np.mean(discounted_sums),
            "std": np.std(discounted_sums)
        }
    # sort the ground truth rewards according to the mean
    ground_truth_rewards = dict(sorted(ground_truth_rewards.items(), key=lambda x: x[1]["mean"], reverse=True))
    #sort_key = []
    # sort the

    sorted_keys = F110Env.eval_agents #['StochasticContinousFTGAgent_0.55_3_0.5_0.03_0.1_5.0_0.3_0.5', 'StochasticContinousFTGAgent_0.7_5_0.5_0.03_0.1_5.0_0.3_0.5', 'pure_pursuit2_0.8_1.0_raceline4_0.3_0.5', 'StochasticContinousFTGAgent_0.85_5_0.5_0.03_0.1_5.0_0.3_0.5', 'pure_pursuit2_0.7_0.7_raceline1_0.3_0.5', 'pure_pursuit2_0.7_0.9_raceline6_0.3_0.5', 'StochasticContinousFTGAgent_0.45_3_0.5_0.03_0.1_5.0_0.3_0.5', 'pure_pursuit2_0.6_0.8_raceline4_0.3_0.5', 'pure_pursuit2_0.7_0.8_raceline2_0.3_0.5', 'pure_pursuit2_0.7_1.0_raceline_0.3_0.5', 'pure_pursuit2_0.7_1.0_raceline1_0.3_0.5', 'pure_pursuit2_0.8_0.8_raceline2_0.3_0.5', 'pure_pursuit2_0.5_0.9_raceline6_0.3_0.5', 'pure_pursuit2_0.9_1.0_raceline4_0.3_0.5', 'pure_pursuit2_0.5_0.75_raceline3_0.3_0.5', 'pure_pursuit2_0.52_0.5_raceline6_0.3_0.5', 'pure_pursuit2_0.9_0.9_raceline6_0.3_0.5', 'pure_pursuit2_0.7_1.1_raceline3_0.3_0.5', 'pure_pursuit2_0.5_0.7_raceline2_0.3_0.5', 'pure_pursuit2_0.5_0.6_raceline3_0.3_0.5', 'pure_pursuit2_0.75_0.9_raceline5_0.3_0.5', 'pure_pursuit2_0.5_0.6_raceline4_0.3_0.5', 'pure_pursuit2_0.7_1.0_raceline2_0.3_0.5', 'pure_pursuit2_0.8_0.85_raceline3_0.3_0.5', 'pure_pursuit2_0.5_0.8_raceline5_0.3_0.5', 'StochasticContinousFTGAgent_0.48_7_0.5_0.03_0.1_5.0_0.3_0.5', 'pure_pursuit2_0.8_1.2_raceline_0.3_0.5', 'pure_pursuit2_0.5_0.5_raceline5_0.3_0.5', 'pure_pursuit2_0.9_1.2_raceline_0.3_0.5', 'pure_pursuit2_0.45_0.4_raceline2_0.3_0.5', 'pure_pursuit2_0.7_1.5_raceline1_0.3_0.5', 'pure_pursuit2_0.43_0.35_raceline6_0.3_0.5', 'pure_pursuit2_0.5_0.4_raceline3_0.3_0.5', 'pure_pursuit2_0.47_0.3_raceline2_0.3_0.5', 'pure_pursuit2_0.5_0.35_raceline5_0.3_0.5', 'StochasticContinousFTGAgent_0.55_10_0.5_0.03_0.1_5.0_0.3_0.5']
    # sort the ground truth rewards according to the sorted keys
    ground_truth_rewards = {k: ground_truth_rewards[k] for k in sorted_keys}
    # print the only the sorted agents as a python array for copy paste
    

    # write the ground truth rewards to a file
    #with open(f"Groundtruth/gt_{args.target_reward}", "w") as f:
    #    json.dump(ground_truth_rewards, f)
    plot_bars_from_dicts([ground_truth_rewards], ["Ground truth"], "Mean discounted reward",plot=False, save_path=f"plots/ground_truth_rewards_{args.target_reward}.pdf")
    print(ground_truth_rewards)


if __name__ == "__main__":
    main(args)
    