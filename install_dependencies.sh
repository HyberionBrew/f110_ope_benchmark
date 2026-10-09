#!/bin/bash

# Create the Conda environment from the environment.yml file

# Initialize Conda for shell interaction
eval "$(conda shell.bash hook)"

# Activate the Conda environment
conda activate f110_ope_benchmark

# Install the required packages
# yamldataclassconfig>=2 breaks track loading in f1tenth_gym ("could not load track Infsaal3")
pip install "yamldataclassconfig==1.5.0"
pip install -e f1tenth_gym/
pip install -e f1tenth_orl_dataset/
pip install -e stochastic_ftg_agents/
pip install -e ope_methods/
