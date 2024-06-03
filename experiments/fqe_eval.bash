#!/bin/bash

# Initial seed value
seed=0

# Number of iterations to run the scripts
iterations=3

# List of dynamics models
# 
fqe_models=( "QFitterL2" "QFitterL2" ) # "ProbsDeltaDynamicsModel" ) "QFitterDD"
agents=( "pure_pursuit2_0.4_0.3_raceline4_0.3_0.5" \ 
    "pure_pursuit2_0.68_1.1_raceline8_0.3_0.5" \ 
    "pure_pursuit2_0.6_1.0_raceline2_0.3_0.5" \ 
    "pure_pursuit2_0.7_0.9_raceline8_0.3_0.5" \ 
    "pure_pursuit2_0.8_0.95_raceline3_0.3_0.5" \ 
    "pure_pursuit2_0.44_0.3_raceline1_0.3_0.5" \ 
    "pure_pursuit2_0.44_0.85_raceline1_0.3_0.5" \
    "pure_pursuit2_0.52_0.9_raceline4_0.3_0.5" \ 
    "pure_pursuit2_0.65_1.2_centerline_0.3_0.5" \  
    "pure_pursuit2_0.73_0.95_centerline_0.3_0.5" \ 
    "pure_pursuit2_0.73_0.95_raceline4_0.3_0.5" \ 
    "StochasticContinousFTGAgent_0.5_2_0.7_0.03_0.1_5.0_0.3_0.5" \ 
    "StochasticContinousFTGAgent_0.6_2_0.8_0.03_0.1_5.0_0.3_0.5" \ 
    "StochasticContinousFTGAgent_0.8_2_0.7_0.03_0.1_5.0_0.3_0.5" \ 
    "StochasticContinousFTGAgent_1.0_1_0.2_0.03_0.1_5.0_0.3_0.5")

lrs=(1e-4)
taus=(0.005)
weight_decays=(1e-5)
# Loop over each dynamics model
for lr in "${lrs[@]}"
    do
    for tau in "${taus[@]}"
    do
        for weight_decay in "${weight_decays[@]}"
        do
            for fqe_model in "${fqe_models[@]}"
            do
                echo "Processing dynamics model: $fqe_model"
                # Loop over each agent
                seed=1
                for ((i=0; i<iterations; i++))
                    do
                    for agent in "${agents[@]}"; do
                    
                    # Reset seed for each dynamics model
                    
                        # if seed is 1 and model is QFitterDD, then skip except if stochastic agent
                        #if [ $seed -eq 1 ] && [ $fqe_model == "QFitterDD" ] && [[ $agent == "pure_pursuit2_0.6_1.0_raceline2_0.3_0.5" ] || [ $agent == "pure_pursuit2_0.68_1.1_raceline8_0.3_0.5" ]]; then #&& [ $agent != "StochasticContinousFTGAgent_0.8_2_0.7_0.03_0.1_5.0_0.3_0.5" ] && [ $agent != "StochasticContinousFTGAgent_1.0_1_0.2_0.03_0.1_5.0_0.3_0.5" ]; then
                        #    continue
                        #fi
                        #if [ "$seed" -eq 1 ] && [ "$fqe_model" = "QFitterDD" ] && { [ "$agent" = "pure_pursuit2_0.6_1.0_raceline2_0.3_0.5" ] || [ "$agent" = "pure_pursuit2_0.68_1.1_raceline8_0.3_0.5" ]; }; then
                        #    continue
                        #fi

                    
                        echo "Iteration $(($i + 1)) with seed $seed for model $fqe_model"


                        # Call reward_rollouts.py
                        python run_fqe.py --target_reward=reward_min_act.json --fqe_model=$fqe_model --seed=$seed --agent=$agent
                        
                        # Increment the seed for the next iteration
                        
                    done
                    seed=$(($seed + 1))
                done
            done
        done
    done
done