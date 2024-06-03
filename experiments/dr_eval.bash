#!/bin/bash

# Define arrays of iw_types, exts, and target_rewards
iw_types=( "step_wis_termination" "phwis_heuristic" "phwis" "step_wis" "simple_is" "simple_step_is" "cobs_wis")
exts=("zero")
rewards=("reward_checkpoint.json" "reward_lifetime.json" "reward_min_act.json" "reward_progress.json" ) # "reward_progress.json" 

# Loop over rewards, iw_types, and exts
for reward in "${rewards[@]}"; do
    for iw_type in "${iw_types[@]}"; do
        for ext in "${exts[@]}"; do
            # Skip the iteration if iw_type is wis_termination since it's commented out
            for seed in {0..0}; do
                echo "Running for reward=$reward, iw_type=$iw_type, ext=$ext"
                python run_iw.py --save --iw_type="$iw_type" --ext="$ext" --target_reward="$reward" --seed="$seed" --dr
                python create_plots_iw_gt.py --ext="$ext" --iw_type="$iw_type" --target_reward="$reward" --seed="$seed" --dr
            done
        done
    done
done