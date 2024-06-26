#!/bin/bash

# Define arrays of iw_types, exts, and target_rewards
iw_types=( "step_wis_termination" "phwis_heuristic" "phwis" "step_wis" "simple_is" "simple_step_is" "cobs_wis")
exts=("zero") # also accepts "mean" and "minus_inf" (this is in log! so zero means prob 1), mean and zero are generally equivalent except for specific cases
rewards=("reward_checkpoint.json" "reward_lifetime.json" "reward_min_act.json" "reward_progress.json" ) # "reward_progress.json" 

# explicitly run for the one mean extension we utilize in the thesis
for reward in "${rewards[@]}"; do
    for seed in {0..0}; do
        python run_iw.py --save --iw_type="step_wis" --ext="mean" --target_reward="$reward" --seed="$seed"
        python create_plots_iw_gt.py --ext="mean" --iw_type="step_wis" --target_reward="$reward" --seed="$seed" 
    done
done
# Loop over rewards, iw_types, and exts
for reward in "${rewards[@]}"; do
    for iw_type in "${iw_types[@]}"; do
        for ext in "${exts[@]}"; do
            # Skip the iteration if iw_type is wis_termination since it's commented out
            for seed in {0..0}; do
                echo "Running for reward=$reward, iw_type=$iw_type, ext=$ext"
                python run_iw.py --save --iw_type="$iw_type" --ext="$ext" --target_reward="$reward" --seed="$seed"
                python create_plots_iw_gt.py --ext="$ext" --iw_type="$iw_type" --target_reward="$reward" --seed="$seed" 
            done
        done
    done
done

