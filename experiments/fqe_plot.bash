#!/bin/bash

# Define the seeds and fitters
seeds=(1 2 3)
fitters=("QFitterL2" "QFitterDD")
python aggregate_fqe.py
# Loop over seeds and fitters
rewards=("reward_checkpoint.json" "reward_lifetime.json" "reward_min_act.json" "reward_progress.json") # "reward_progress.json" 

# Loop over rewards, iw_types, and exts
for reward in "${rewards[@]}"; do
    for seed in "${seeds[@]}"; do
        for fitter in "${fitters[@]}"; do
            echo "Running with seed=${seed} and fitter=${fitter} reward=${reward}"
            python create_plots_fqe_gt.py --seed=${seed} --fitter=${fitter} --target_reward=${reward}
        done
    done
done