#!/bin/bash

# Create the Conda environment from the environment.yml file

# Initialize Conda for shell interaction
eval "$(conda shell.bash hook)"

# Activate the Conda environment
conda activate f110_ope_benchmark

# Install the required packages
pip install -e f1tenth_gym/
pip install -e f1tenth_orl_dataset/
pip install -e stochastic_ftg_agents/
pip install -e ope_methods/
