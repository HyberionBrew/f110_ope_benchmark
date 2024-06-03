import os
import json
import numpy as np
import matplotlib.pyplot as plt
# List of DynamicsNetwork names
import seaborn as sns
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
})

dynamics_networks4 = ["Simulation","SimpleDynamicsModel","ProbDynamicsModel","DeltaDynamicsModel","AutoregressiveModel", "ProbsDeltaDynamicsModel","AutoregressiveDeltaModel", "EnsembleARDModel"][::-1]#"EnsemblePDDModel", "EnsembleSDModel", "EnsembleARModel", "EnsembleARDModel"] #,"ProbsDeltaDynamicsModel", "AutoregressiveModel", "EnsemblePDDModel", "NormImportanceSampling"]  # Replace with your actual network names
dynamics_networks4 = ["EnsembleARDModel", "EnsemblePDDModel"]
#dynamics_networks += ["EnsembleSDModel", "EnsembleARModel", "EnsemblePDDModel", "EnsembleARDModel"]
# take all folder names from:
#dir_iw = "/home/fabian/msc/f110_dope/ws_release/experiments/runs_iw3"
#dynamics_networks= [d for d in os.listdir(dir_iw) if os.path.isdir(os.path.join(dir_iw, d))]
#dynamics_networks = ['iw_action_step_wis_termination_mean', 'iw_action_step_wis_mean', 'iw_action_step_wis_zero','iw_action_phwis_zero',
#                      'iw_action_cobs_wis_zero', 
#                      'iw_action_simple_step_is_zero',  
#                     'iw_action_simple_is_mean',
#                      ]
dynamics_networks1 = ['iw_action_step_wis_termination_zero_dr', 'iw_action_cobs_wis_zero_dr', 'iw_action_simple_step_is_zero_dr','iw_action_simple_is_zero_dr']
#                      'iw_action_cobs_wis_zero', 
#                      'iw_action_simple_step_is_zero',  
#                     'iw_action_simple_is_mean',
#                      ]
#dynamics_networks1 = ['iw_action_step_wis_termination_zero_dr', 'iw_action_simple_step_is_zero_dr']
dynamics_networks2 = ['iw_action_step_wis_termination_zero', 'iw_action_step_wis_mean','iw_action_step_wis_zero',  'iw_action_cobs_wis_zero', 'iw_action_simple_step_is_zero','iw_action_simple_is_zero',
                      'iw_action_phwis_zero',
                      ]
dynamics_networks2 = ['iw_action_step_wis_termination_zero', 'iw_action_cobs_wis_zero', 'iw_action_simple_step_is_zero', 'iw_action_simple_is_zero']
# label_map = ["Simulation", "Naive Model (MSE)", "Naive Model (LL)", "Delta Model (MSE)", "Delta Model (LL)", "Autoregressive Model", "Autoregressive Delta Model"][::-1]
dynamics_networks3 = ["QFitterDD", "QFitterL2"]
#dynamics_networks3 = ["QFitterDD"]
label_map4 = ["Simulation", "NM (MSE)", "NM (LL)", "DM (MSE)",  "AM", "DM (LL)", "ADM", "ADM (5)"][::-1]
label_map4 = ["ADM (5)", "DM (5)"]
label_map1 = ["TPDWIS (DR)", "WIS (DR)", "PDIS (DR)", "IS (DR)"]
#label_map1 = ["TDPWIS (DR)", "PDIS (DR)"]
label_map2 = ["TPDWIS","PDWIS (Mean)", "PDWIS (C)", "WIS", "PDIS", "IS", "PHWIS" ]
label_map2 = ["TPDWIS"]
label_map2 = ["TPDWIS", "WIS", "PDIS", "IS"]
label_map3 = ["FQE (DD)", "FQE (L2)"]
#label_map3 = ["FQE (DD)"]
filter  = ["iw_action_step_wis_termination_zero_dr",
           "iw_action_step_wis_termination_zero",
           "QFitterDD",
           "AutoregressiveDeltaModel"]
#label_map = ["NM (MSE)", "DM (LL)", "AM", "ADM"][::-1]
#all_labels = ["Simulation", "NM (MSE)", "NM (LL)", "DM (MSE)", "DM (LL)", "AM", "ADM"][::-1] + ["EnsembleSDModel", "EnsembleARModel", "EnsemblePDDModel", "EnsembleARDModel"]
#label_map = ["TPDWIS", "PDWIS (Mean)", "PDWIS (Ones)", "PHWIS", "WIS", "PDIS", "IS"]
# TPDWIS, PDWIS (Mean) ,PDWIS (Ones), PHWIS, WIS, PDIS, IS
#print(dynamics_networks)
#dynamics_networks = ["iw_action_cobs_wis_mean", "iw_action_step_wis_termination_mean", "iw_action_simple_step_is_mean"]
max_seed = 25 # Important to change!
base_path = "runs_mb"
#base_path= "runs_iw3"
base_path1 = "runs_dr"
base_path3 = "fqe_all"
base_path2 = "runs_iw3"
base_path4 = "runs_mb"
dynamics_networks_all = [ dynamics_networks1, dynamics_networks2, dynamics_networks3, dynamics_networks4]
base_paths_all = [base_path1, base_path2, base_path3, base_path4]
label_maps_all = [label_map1, label_map2, label_map3, label_map4]
names = ["DR", "IS", "FQE", "MB"]
def calculate_stats(metrics):
    stats = {key: {"mean": np.mean(values), "std": np.std(values)} for key, values in metrics.items()}
    stats["regret@1"] = {"mean": np.mean(metrics["regret@1"]), "std": np.std(metrics["regret@1"])}
    return stats

def plot_grouped_bars(data_dict, title="Grouped Data Comparison", metric="spearman_corr"):
    """
    Plot grouped bar charts with mean and standard deviation, handling varying subkeys.

    Args:
        data_dict (dict): Dictionary with a variable structure of subkeys, like:
                          {
                              'Group1': {'Subkey1': {'spearman_corr': {'mean': val1, 'std': std1}}, ...},
                              'Group2': {'Subkey3': {'spearman_corr': {'mean': val2, 'std': std2}}, ...},
                              ...
                          }
        title (str): Title of the plot.
        metric (str): The metric to plot ('spearman_corr' by default).

    Returns:
        None: Displays the plot.
    """
    # Collecting all unique subkeys across all groups
    #subkeys = set()
    #for group_data in data_dict.values():
    #    subkeys.update(group_data.keys())
    #subkeys = sorted(subkeys)  # Sort to maintain consistent order

    #groups = list(data_dict.keys())
    #n_groups = len(groups)
    #n_subkeys = len(subkeys)

    fig, ax = plt.subplots(figsize=(12, 6))
    #ind = np.arange(n_subkeys)  # the x locations for the groups
    width = 1.0   # the width of the bars

    # Creating bar plots for each group, only where data is available
    prev_group_sizes = 0 # keep track for positioning
    all_subkeys = []
    x_ticks_pos = []
    for i, group in enumerate(data_dict):
        print(group)
        print('--------')
        #if group != "DR":
        #    continue
        group_data = data_dict[group]
        group_means = []
        group_errors = []
        group_indices = []

        for subkey in group_data:
            group_means.append(group_data[subkey][metric]['mean'])
            group_errors.append(group_data[subkey][metric]['std'])
                #group_indices.append(j)
        print(group)
        print(len(data_dict[group].keys()))
        print(len(group_means))
        print(prev_group_sizes * i)
        current_positions = np.arange(len(data_dict[group].keys())) + prev_group_sizes
        ax.bar(np.arange(len(data_dict[group].keys())) + prev_group_sizes, group_means, 0.75, label=group, yerr=group_errors, capsize=5, color=sns.color_palette()[i])
        x_ticks_pos += list(current_positions)
        prev_group_sizes += len(group_means) * width + width
        all_subkeys += list(group_data.keys())
        
    
    print(title)
    if title=="Min_act Reward":
        ax.set_title("Minimum Action Reward", fontsize = 30)
    else:
        ax.set_title(title, fontsize = 30)
    ax.set_xticks(x_ticks_pos)
    ax.set_xticklabels(all_subkeys)
    if metric == "spearman_corr":
        ax.set_ylabel("Spearman Rank Correlation", fontsize=25)
    elif metric == "abs":
        ax.set_ylabel("Mean Absolute Error", fontsize=25)
    elif metric == "regret@1":
        ax.set_ylabel("Regret@1", fontsize=25)
    #ax.set_ylabel("Spearman Rank Correlation", fontsize=25)
    # set y tick fontsize
    ax.tick_params(axis='y', labelsize=25)
    #ax.legend()
    # set legend to left bottom corner an adjust fontsize
    #ax.legend(fontsize=23, loc='lower left')
    # increase y ticks fontsize
    # set maximum y limit to 1
    if metric == "spearman_corr":
        ax.set_ylim(-0.5, 1)
    
    if metric == "abs":
        # make y axis logarithmic
        plt.yscale('log')
        plt.ylim(0, 100)
    elif metric == "regret@1":
        plt.ylim(0, 30)

    plt.xticks(rotation=45, fontsize=25)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    # strip all whitespaces from metric and title
    metric = metric.replace(" ", "")
    title = title.replace(" ", "")
    plt.savefig(f"plots/0_{metric}_cross_{title}.pdf")
    plt.show()
  



def plot_stats_comparison(stats1, stats2, stats3, metric, title="Comparison of Mean and STD of Metric", save_path=None):
    # labels = dynamics_networks  # List of labels assumed to be defined elsewhere
    labels_1 = list(stats1.keys())
    labels_2 = list(stats2.keys())
    labels_3 = list(stats3.keys())
    # 3 FQE just take FQEDD
    print(stats1)
    print(stats2)
    print(stats3)

    means1 = [stats1[label][metric]['mean'] for label in labels_1]
    errors1 = [stats1[label][metric]['std'] for label in labels_1]
    means2 = [stats2[label][metric]['mean'] for label in labels_2]
    errors2 = [stats2[label][metric]['std'] for label in labels_2]
    means3 = [stats3[label][metric]['mean'] for label in labels_3]
    errors3 = [stats3[label][metric]['std'] for label in labels_3]
    
    # Simplifying label names by removing "Model"
    labels = [label.replace("Model", "") for label in labels_1]
    
    sns.set_theme(style="white")
    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.25  # Width of the bars
    
    # Plotting
    fqe_mean = [ stats3["FQE (DD)"][metric]['mean']] * len(labels_1)
    fqe_error = [ stats3["FQE (DD)"][metric]['std']] * len(labels_1)
    dr_mean = [ stats1[label][metric]['mean'] for label in labels_1]
    dr_error = [ stats1[label][metric]['std'] for label in labels_1]
    print(labels_1)
    print(labels_2)
    iw_mean = [ stats2[label][metric]['mean'] for label in labels_2]
    iw_error = [ stats2[label][metric]['std'] for label in labels_2]
    print(iw_mean)

    plt.bar(x - 0.25, dr_mean, width, label='Stats1 Mean', yerr=dr_error, capsize=5, color='skyblue')
    plt.bar(x + 0, iw_mean, width, label='Stats2 Mean', yerr=iw_error, capsize=5, color='darkorange') #sns.color_palette()[1])
    plt.bar(x + 0.25,fqe_mean, width, label='Stats3 Mean', yerr=fqe_error, capsize=5, color='green')
    # plt.xlabel('Dynamics Network', fontsize=20)
    if metric == "spearman_corr":
        plt.ylabel("Spearman Correlation", fontsize=20)
    elif metric == "abs":
        plt.ylabel("Mean Absolute Error", fontsize=20)
    elif metric == "regret@1":
        plt.ylabel("Regret@1", fontsize=20)
    plt.title(title, fontsize=22, pad=30)
    plt.xticks(x, label_map1, rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(axis='y')
    
    # Adjusting the y-axis limits based on the metric
    if metric == "spearman_corr":
        plt.ylim(-0.5, 1)
    if metric == "abs":
        # make y axis logarithmic
        plt.yscale('log')
        plt.ylim(0, 100)
    elif metric == "regret@1":
        plt.ylim(0, 30)
    # overwrite legend with 1) Naive, 2) Ensemble
    # set legend to the bottom right corner
    # plot the legend outside of the plot

    plt.legend(["DR", "IS", "FQE (DD)"], fontsize=20, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

    # Adjust layout to make room for the legend
    plt.subplots_adjust(top=0.75)  
        
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_stats(stats, metric, labels, title="Mean and STD of metric", save_path=None):
    labels = labels #["SimpleDynamicsModel","ProbDynamicsModel","DeltaDynamicsModel","ProbsDeltaDynamicsModel", "AutoregressiveModel"]#list(stats.keys())
    print(labels)
    print(stats)
    # unroll loop below to find error
    for label in labels:
        print(label)
        print(stats[label][metric]['mean'])
        print(stats[label][metric]['std'])
    print(';;;;;')
    means = [stats[label][metric]['mean'] for label in labels]
    errors = [stats[label][metric]['std'] for label in labels]
    # from each label remove the Model part
    labels = [label.replace("Model", "") for label in labels]
    sns.set_theme(style="white")
    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.35
    
    #fig, ax = plt.subplots()
    plt.bar(x - width/2, means, width, label='Mean', yerr=errors, capsize=5)
    # plt.xlabel('Dynamics Network', fontsize=20)
    if metric == "spearman_corr":
        plt.ylabel("Spearman Correlation", fontsize=20)
    if metric == "abs":
        plt.ylabel("mean-absolute error", fontsize=20)
    #plt.ylabel(metric, fontsize=20)
    plt.title(title, fontsize=22)
    plt.xticks(x-width/4, names, rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    # add grid x direction
    plt.grid(axis='y')
    # fix the y axis between -0.3 and 1
    if metric == "spearman_corr":
        plt.ylim(-0.5, 1)
    if metric == "abs":
        # make y axis logarithmic
        plt.yscale('log')
        plt.ylim(0, 100)
    elif metric == "regret@1":
        plt.ylim(0, 30)
        #plt.ylim(0, 50)
    #ax.set_ylabel('Scores')
    #ax.set_title(title)
    #ax.set_xticks(x)
    #ax.set_xticklabels(labels, rotation=45)
    #ax.legend()
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    exit()
def plot_comparison_ref_3(stats_list, metric, title="Comparison of Mean and STD of Metric", save_path=None):
    # Assuming all dictionaries in stats_list have the same keys in the same order
    labels = list(stats_list[0].keys())  # Get keys from the first dictionary
    num_methods = len(labels)  # Number of methods (like "DR", "IS", etc.)
    num_groups = len(stats_list)  # Number of groups
    width = 0.8 / num_methods  # Width of the bars, dynamically adjusted
    
    # Set up the plot
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="white")
    x = np.arange(num_groups)  # Label locations for groups
    
    # Plotting
    for j, label in enumerate(labels):
        means = [stats[label][metric]['mean'] for stats in stats_list]
        errors = [stats[label][metric]['std'] for stats in stats_list]
        plt.bar(x + j * width, means, width, label=label, yerr=errors, capsize=5)

    # Setting labels and title
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=20)
    plt.title(title, fontsize=23)
    plt.xticks(x + width * (num_methods - 1) / 2,["Progress", "Checkpoint", "Lifetime", "Minimum Action"], rotation=0, fontsize=20)  # Centering the x-ticks
    # set size of ytick text
    plt.yticks(fontsize=18)

    plt.grid(axis='y')

    # Dynamically setting y-axis limits if desired based on metric
    if metric == "spearman_corr":
        plt.ylim(-0.5, 1)
    elif metric == "abs":
        plt.ylim(0, None)  # Automatically adjust y-limits
    elif metric == "regret@1":
        plt.ylim(0, None)  # Automatically adjust y-limits

    plt.legend(fontsize=23, loc='lower right')
    plt.tight_layout()

    # Saving or showing the plot
    if save_path:
        plt.savefig(save_path)
    plt.show()
def plot_comparison_ref_2(stats_list, metric, title="Comparison of Mean and STD of Metric", save_path=None):
    # Assuming all dictionaries in stats_list have the same keys in the same order
    labels = list(stats_list[0].keys())
    num_groups = len(labels)
    num_stats = len(stats_list)
    width = 0.8 / num_stats  # Width of the bars, dynamically adjusted
    
    # Set up the plot
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="white")
    x = np.arange(num_groups)  # Label locations
    
    # Plotting
    for i, stats in enumerate(stats_list):
        means = [stats[label][metric]['mean'] for label in labels]
        errors = [stats[label][metric]['std'] for label in labels]
        plt.bar(x + i * width, means, width, label=f'Stats from {i+1}', yerr=errors, capsize=5)

    # Setting labels and title
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(x + width * (num_stats - 1) / 2, labels, rotation=45)  # Centering the x-ticks
    plt.grid(axis='y')

    # Dynamically setting y-axis limits if desired based on metric
    if metric == "spearman_corr":
        plt.ylim(-0.5, 1)
    elif metric == "abs":
        plt.ylim(0, None)  # Automatically adjust y-limits
    elif metric == "regret@1":
        plt.ylim(0, None)  # Automatically adjust y-limits

    plt.legend(title="Data Source", fontsize=12, loc='upper right')
    plt.tight_layout()

    # Saving or showing the plot
    if save_path:
        plt.savefig(save_path)
    plt.show()
def plot_comparison_ref(stats_list, reward_names, metric, title="Comparison of Mean and STD of Metric", save_path=None):
    # Extract labels and corresponding mean and std values
    labels = list(stats_list[0].keys())  # Assume all stats have the same structure
    x = np.arange(len(labels))
    width = 0.2  # width of the bars
    
    # Set figure size and style
    plt.figure(figsize=(15, 7))
    sns.set_theme(style="white")
    
    # Data gathering for plot
    means = []
    errors = []
    for stats in stats_list:
        means.append([stats[method][metric]['mean'] for method in labels])
        errors.append([stats[method][metric]['std'] for method in labels])
    
    # Plotting
    for i, (mean, error) in enumerate(zip(means, errors)):
        plt.bar(x + i * width - width, mean, width, label=f'Stats {i+1} Mean', yerr=error, capsize=5)

    # Set labels and title
    plt.ylabel(metric.replace('_', ' ').title() + ' Value', fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(x, reward_names, rotation=45)
    plt.grid(axis='y')
    
    # Adjusting y-axis limits dynamically based on metric types
    if metric == "spearman_corr":
        plt.ylim(-0.5, 1)
    elif metric == "abs":
        plt.ylim(0, max(max(means) + max(errors)) * 1.1)  # Dynamic limit based on max mean + error
    elif metric == "regret@1":
        plt.ylim(0, max(max(means) + max(errors)) * 1.1)

    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()

    # Save or show plot
    if save_path:
        plt.savefig(save_path)
    plt.show()


def read_metrics_from_json(network_name, reward_name="reward_progress"):
    network_path = os.path.join(base_path, network_name, "f110-real-stoch-v2/250/off-policy")
    #network_path = os.path.join("/home/fabian/msc/f110_dope/ws_release/experiments/runs_iw3", network_name, "f110-real-stoch-v2/250/off-policy")
    results = {"spearman_corr": [], "abs": [], "regret@1": []}
    file_not_found_counter = 0 # Counter for files not found
    if os.path.exists(network_path):
        seeds = [d for d in os.listdir(network_path) if os.path.isdir(os.path.join(network_path, d))]
        for seed in seeds:
            if int(seed) > max_seed:
                continue
            json_path = os.path.join(network_path, seed, "results", f"reward_{reward_name}._metrics.json")
            if os.path.isfile(json_path):
                if network_name == "iw":
                    print("??")
                with open(json_path, 'r') as f:
                    print(json_path)
                    data = json.load(f)
                    if 'spearman_corr' in data and 'abs' in data:
                        results['spearman_corr'].append(data['spearman_corr'])
                        results['abs'].append(data['abs'])
            else:
                print(f"File not found: {json_path}")
                file_not_found_counter += 1
            # now read the specific value and compute the regret@1
            # load the ground truth reward
            
            ground_truth_rewards_folder = f"../Groundtruth/gt_{reward_name}.json"
            with open(ground_truth_rewards_folder, "r") as f:
                ground_truth_rewards = json.load(f)
            # get the max from the ground truth rewards
            max_reward = max([ground_truth_rewards[agent]['mean'] for agent in ground_truth_rewards])
            json_path = os.path.join(network_path, seed, "results", f"{reward_name}.json")
            if os.path.isfile(json_path):
                with open(json_path, 'r') as f:
                    print(json_path)
                    data = json.load(f)
                    max_agent = None
                    for agent in data:
                        if max_agent is None or data[agent]['mean'] > data[max_agent]['mean']:
                            max_agent = agent
                    #max_reward_agent = max([data[agent]['mean'] for agent in data])
                    print(max_reward - ground_truth_rewards[max_agent]['mean'])
                    results['regret@1'].append(abs(max_reward - ground_truth_rewards[max_agent]['mean']))
            else:
                print(f"File not found: {json_path}")
                file_not_found_counter += 1
            print(f"Regret@1: {results['regret@1']}")
    else:
        print(f"Path does not exist: {network_path}")
    
    return results, file_not_found_counter

# Iterate through each dynamics network and print results
list_for_comp = []
reward_order =["reward_progress", "reward_checkpoint", "reward_lifetime", "reward_min_act"]
for reward in reward_order:
    all_stats = {}
    for name, dynamics_networks, label_map, base_path in zip(names, dynamics_networks_all, label_maps_all, base_paths_all):
        #if name == "DR":
        #    pass
            #continue
        print(name)
        all_stats[name] = {}
        for network_name, label in zip(dynamics_networks, label_map):
            metrics, files_not_founds = read_metrics_from_json(network_name, reward_name=reward)
            #print(metrics)

            stats = calculate_stats(metrics)
            all_stats[name][label] = stats
            #print(f"Results for {network_name}:")
            #print(f"Spearman Correlation: {metrics['spearman_corr']}")
            #print(f"Absolute values: {metrics['abs']}\n")
            #print(f"Regret@1: {metrics['regret@1']}\n")

        if name == "IW":
            print(all_stats[name])

    print(all_stats)
    # loop over keys and count items in each key
    for key in all_stats:
        print(key)
        print(len(all_stats[key]))
    print(all_stats.keys())
    #plot_grouped_bars(all_stats, title=f"{reward[7].upper()}{reward[8:]} Reward", metric="spearman_corr")
    #plot_grouped_bars(all_stats, title=f"{reward[7].upper()}{reward[8:]} Reward", metric="abs")
    #plot_grouped_bars(all_stats, title=f"{reward[7].upper()}{reward[8:]} Reward", metric="regret@1")
    #continue
    #exit()
    # TODO! add to thesis the below plots
    title = f"{reward[7].upper()}{reward[8:]} Reward"
    if title == "Min_act Reward":
        title = "Minimum Action Reward"
    plot_stats_comparison(all_stats["DR"],  all_stats["IS"],
                           all_stats["FQE"],
                          "spearman_corr", title=title, 
                          save_path=f"plots/00_spearman_corr_comparison_{reward}_work.pdf")
    plot_stats_comparison(all_stats["DR"],  all_stats["IS"],
                        all_stats["FQE"],
                        "abs", title=title, 
                        save_path=f"plots/00_abs_{reward}_work.pdf")
    plot_stats_comparison(all_stats["DR"],  all_stats["IS"],
                        all_stats["FQE"],
                        "regret@1", title=title, 
                        save_path=f"plots/00_regret_{reward}_work.pdf")
    continue
    # from all_stats filter out only the stats we want
    filtered_stats = {}
    print(names)
    #filter  = ["iw_action_step_wis_termination_zero_dr",
    #       "iw_action_step_wis_termination_zero",
    #       "QFitterDD",
    #       "AutoregressiveDeltaModel"]
    #for name, filter in zip(names, filter):
    #filtered_stats[name] = {}
    #filtered_stats[name] = all_stats[name].copy()
        #filtered_stats[name][filter] = 
    print('------')
    print(filtered_stats)
    print('------')
    plot_stats(filtered_stats, "spearman_corr",names, title=f" {reward[7:]}-reward \n Spearman Correlation", save_path=f"plots/spearman_corr_{reward}.pdf")
    #plot_stats(filtered_stats, "abs",names, title=f" {reward[7:]}-reward \nAbsolute error", save_path=f"plots/abs_{reward}.pdf")
    #plot_stats(filtered_stats, "regret@1",names, title=f" {reward[7:]}-reward \n Regret@1", save_path=f"plots/regret@1_{reward}.pdf")
    list_for_comp.append(all_stats.copy())

    # write the metrics into a nice latex file

    #with open(f"{base_path}/metrics_{reward}.txt", "w") as f:
    #    for i, model in enumerate(dynamics_networks):
            #f.write(f"{label_map[i]} & {all_stats[model]['spearman_corr']['mean']:.2f} & {all_stats[model]['spearman_corr']['std']:.2f} & {all_stats[model]['abs']['mean']:.2f} & {all_stats[model]['abs']['std']:.2f} & {all_stats[model]['regret@1']['mean']:.2f} & {all_stats[model]['regret@1']['std']:.2f} \\\\ \n")
    #        f.write(f"& {label_map[i]} & {all_stats[model]['spearman_corr']['mean']:.2f} $\pm$ {all_stats[model]['spearman_corr']['std']:.2f}  & {all_stats[model]['abs']['mean']:.2f} $\pm$ {all_stats[model]['abs']['std']:.2f}  & {all_stats[model]['regret@1']['mean']:.2f} $\pm$ {all_stats[model]['regret@1']['std']:.2f} \\\\ \n")
    """
    models = ["SimpleDynamicsModel","ProbsDeltaDynamicsModel", "AutoregressiveModel" ,"AutoregressiveDeltaModel"][::-1]

    ensembles = ["EnsembleSDModel",  "EnsemblePDDModel", "EnsembleARModel", "EnsembleARDModel"][::-1]
    all_stats_models = {model: all_stats[model] for model in models}
    all_stats_ensembles = {model: all_stats[model] for model in ensembles}
    
    plot_stats_comparison(all_stats_models
                          , all_stats_ensembles, 
                          "spearman_corr", title=f" {reward[7:]}-reward \n Spearman Correlation", 
                          save_path=f"plots/spearman_corr_comparison_{reward}.pdf")
    # abs
    plot_stats_comparison(all_stats_models
                          , all_stats_ensembles, 
                          "abs", title=f" {reward[7:]}-reward \n Absolute Error", 
                          save_path=f"plots/abs_comparison_{reward}.pdf")
    # regret@1
    plot_stats_comparison(all_stats_models
                          , all_stats_ensembles, 
                          "regret@1", title=f" {reward[7:]}-reward \n Regret@1", 
                          save_path=f"plots/regret@1_comparison_{reward}.pdf")
    """
    # print files not found with warning if > 0
    if files_not_founds > 0:
        print(f"Warning: {files_not_founds} files not found")
    else:
        print(f"All files found for {reward}")

for i in list_for_comp:
    print(i)
    print(len(i))
plot_comparison_ref_3(list_for_comp, "spearman_corr", title=f" Spearman Correlation", 
                        save_path=f"plots/spearman_corr_comparison_{reward}.pdf")