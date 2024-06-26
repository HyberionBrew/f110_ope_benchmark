import os
import json
rewards = ["reward_min_act.json", "reward_progress.json", "reward_checkpoint.json", "reward_lifetime.json"]
def aggregate_results(directory, reward):
    # Traverse the user-provided directory to find all seed subfolders
    for seed_folder in os.listdir(directory):
        seed_path = os.path.join(directory, seed_folder)
        if os.path.isdir(seed_path):
            # Initialize the aggregated results dictionary
            aggregated_results = {}
            
            # For each seed subfolder, look for agent subfolders
            for agent_folder in os.listdir(seed_path):
                agent_path = os.path.join(seed_path, agent_folder)
                if os.path.isdir(agent_path):
                    results_path = os.path.join(agent_path, "results")
                    if os.path.exists(results_path) and os.path.isdir(results_path):
                        # Assuming there's only one .json file in the results folder
                        for file in os.listdir(results_path):
                            if file.endswith(".json"):
                                with open(os.path.join(results_path, file), 'r') as f:
                                    # Load the dictionary from the JSON file
                                    agent_results = json.load(f)
                                    # Save the dictionary, keyed by the agent name
                                    aggregated_results[agent_folder] = agent_results[agent_folder]
            
            # After processing all agents for the seed, save the aggregated dictionary
            if aggregated_results:
                print(f"Saving aggregated results for seed {seed_folder} to {seed_path}.")
                with open(os.path.join(seed_path,reward), 'w') as f:
                    json.dump(aggregated_results, f, indent=4)


# Replace "your_directory_path_here" with the path to the directory containing the seed subfolders

for reward in rewards:
    directory_path = f"runs_fqe_0.0001_0.005_1e-05_{reward}/QFitterDD/f110-real-stoch-v2/250/on-policy"
    aggregate_results(directory_path, reward)
    directory_path = f"runs_fqe_0.0001_0.005_1e-05_{reward}/QFitterL2/f110-real-stoch-v2/250/on-policy"
    aggregate_results(directory_path, reward)

print("Aggregation complete.")