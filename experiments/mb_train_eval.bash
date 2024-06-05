#!/bin/bash

# Initial seed value
seed=0

# Number of iterations to run the scripts
iterations=25

# List of dynamics models
# 
dynamics_models=("AutoregressiveDeltaModel" "SimpleDynamicsModel" "DeltaDynamicsModel" "ProbDynamicsModel" "ProbsDeltaDynamicsModel" "AutoregressiveModel" "EnsembleSDModel" "EnsemblePDDModel" "EnsembleARModel" "EnsembleARDModel") #  #( "DeltaDynamicsModel" ) # "ProbsDeltaDynamicsModel" )

# Loop over each dynamics model
for dynamics_model in "${dynamics_models[@]}"
do
    echo "Processing dynamics model: $dynamics_model"

    # Reset seed for each dynamics model
    seed=0

    for ((i=0; i<iterations; i++))
    do
        echo "Iteration $(($i + 1)) with seed $seed for model $dynamics_model"

        # Call train_mb.py (commented out, but included for reference)

        # Call reward_rollouts.py
        python train_mb.py --dynamics_model=$dynamics_model --split=off-policy --train --save_model --update_steps=10_001 --seed=$seed --skip_eval
        # 
        python reward_rollouts.py --dynamics_model=$dynamics_model --split=off-policy --model_checkpoint=model_${seed}_10000.pth --target_reward reward_progress.json reward_lifetime.json reward_checkpoint.json reward_min_act.json --save --seed=$seed --rollouts_per_initial_state=1
        #python reward_rollouts.py --dynamics_model=$dynamics_model --split=off-policy --model_checkpoint=model_${seed}_10000.pth --target_reward reward_progress.json --save --seed=$seed --rollouts_per_initial_state=1
        python create_plots.py --dynamics_model=$dynamics_model  --target_reward=reward_lifetime.json --seed=$seed
        python create_plots.py --dynamics_model=$dynamics_model  --target_reward=reward_checkpoint.json --seed=$seed
        python create_plots.py --dynamics_model=$dynamics_model  --target_reward=reward_progress.json --seed=$seed
        python create_plots.py --dynamics_model=$dynamics_model  --target_reward=reward_min_act.json --seed=$seed
        
        # Increment the seed for the next iteration
        seed=$(($seed + 1))
    done
done
