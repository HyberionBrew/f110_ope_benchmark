#!/bin/bash

# Create the Conda environment from the environment.yml file
#conda env create -f environment.yml

# Initialize Conda for shell interaction
eval "$(conda shell.bash hook)"

# Activate the Conda environment
conda activate f110_ope_benchmark

# Install the required packages
pip install f1tenth_gym/
pip install f1tenth_orl_dataset/
pip install stochastic_ftg_agents/
pip install ope_methods/