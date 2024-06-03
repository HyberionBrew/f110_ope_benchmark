Code to reproduce the real-world F110 OPE Benchmark.

# Installation

First, initialize and pull the submodules, setting them to their correct branches (does this actually work, todo check!).
```
git submodule update --init --recursive --remote
```
Setup the conda environment.
```
conda env create -f environment.yml
```

TODO! Add something on fetching the dataset

Checkout the conda environment
```
conda activate f110_ope_benchmark
```


```
pip install f1tenth_gym/
```

```
pip install f1tenth_orl_dataset/
```

```
pip install stochastic_ftg_agents/
```

```
pip install ope_methods/
```


# Run the experiments
Go into the experiments folder for all the following steps.

```
cd experiments
```

## Fitted-Q-Evaluation

In order to run FQE experiments, we first need to train the FQE models.
Executing the train script will train all 15 eval agents, with 3 different seeds for one target reward - the target reward has to be manually adjusted in the bash script.
This will take a very long time  (multiple days on RTX 3090)! In the bash script it is possible to manually adapt the maximum number of update steps and amount of seeds:

```
bash fqe_train.bash
```

After training running evaluation:

```
bash fqe_eval.bash
```

In case the number of update steps has been changed a manual change in run_fqe is required, where the fqe model is loaded.

Results + Intermediate models are placed into 'fqe_runs_\[param-string\]'.

Per agent estimation results are made available in a folder called 'results' at the bottom of the respective agents folder hierarchy.
As these are per-agent results we still need to aggregate them, such that we have per-seed results. To do so modify and run:

```
python aggregate_fqe.py
```

for each reward you are interested in.


## Model-Based

Model-based OPE training and evaluation can be run with the following command:

```
bash mb_train_eval.bash
```

Results are placed into 'runs_mb'. Some visualizations and the final results are also available in that folder.

## Importance Sampling

In order to run IS:

```
bash iw_eval.bash
```

## Doubly Robust

First, manually change the path to the fqe model in line 216 run_iw.py (just have to change once to the absolute location of fqe_runs).

Then execute:

```
bash dr_eval.bash
```


# Plotting and Visualization

The previous steps result in some visualizations and seed-wise results. In order to aggregate the results and generate overall plots and statistics, some sample code is provided in 'plot_results.py'. However, be aware that this code is under construction.

# TODOs

1) Remove all the absolute paths
2) Forward the submodules
3) Redo all plots one more time, after feedback (very small bug, affecting the results minusculy)
4) Add instructions on adding the dataset path
5) Rework the plotting step asap
6) Add existing data to be downloaded such that plotting can be done immediately without running all the other scripts
7) Do one full trial run with IS, MB (not all models), FQE (only 1 seed and one agent), IS (DR) in a docker enviroment.
8) Provide code on generating the gts?
9) Update the description of the F1tenth dataset

10) 
