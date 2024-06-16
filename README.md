Code to reproduce the real-world F110 OPE Benchmark.
# Installation with Docker

```
git clone https://github.com/HyberionBrew/f110_ope_benchmark.git
cd f110_ope_benchmark/docker
docker build -t ubuntu-cuda:22.04.3 .
```


Modify the local volume to point to you local experiments folder (needs to be this repo!).
```
docker run --gpus all -it -v /home/fabian/msc/f110_dope/f110_ope_benchmark/experiments:/f110_ope_benchmark/experiments ubuntu-cuda:22.04.3
```

In the docker environment:
```
conda activate f110_ope_benchmark
cd f110_ope_benchmark/experiments/
```
Now we are setup to run any of the below experiments!

# Run the experiments
Go into the experiments folder (in docker) for all the following steps.

```
cd experiments
```

All output files will be available on the host machine in the experiments folder.
The output products are names runs_mb, runs_iw and so on.

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

The output artifacts we generated for FQE can also be downloaded like this (skipping the two above steps):

```
gdown --id 1Rv0mZOQwh3HDDNW4ZveaMHaHxfPVYi8a -O /f110_ope_benchmark/experiments/runs_fqe.zip
unzip /f110_ope_benchmark/experiments/runs_fqe.zip -d /f110_ope_benchmark/experiments/
```


In case the number of update steps has been changed a manual change in run_fqe is required, where the fqe model is loaded.

Results + Intermediate models are placed into 'fqe_runs_\[param-string\]'.

Per agent estimation, results are made available in a folder called 'results' at the bottom of the respective agent's folder hierarchy.
As these are per-agent results we still need to aggregate them, such that we have per-seed results. To do so run:

```
bash fqe_plot.bash
```

## Doubly Robust

For DR we need FQE models. These are included if you chose to download the fqe artifacts. 

Manually change the path to the fqe model in line 216 run_iw.py, if the  (just have to change once to the absolute location of fqe_runs).


Then execute:

```
bash dr_eval.bash
```


# Plotting and Visualization

Take a look at the notebook: `final_results.ipynb`
