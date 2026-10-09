Code to reproduce the real-world F110 OPE Benchmark.

# Known issue: batched pure pursuit agents (fixed in October 2026)

The pure pursuit agents in `stochastic_ftg_agents` computed the lookahead point wrongly when called on a batch of states: every state in a batch used the lookahead offset of the first state of the batch. Single states, as used on the car and in the simulator while recording the datasets, were not affected, so the datasets, their `log_probs` and the ground truth are correct. All OPE methods (IS, DR, FQE and model-based) evaluate the target policies in batches, so the estimates for the 11 pure pursuit target policies in the original results used slightly wrong target actions and target log-probabilities (steering deltas off by a median of 0-0.003 rad and a 90th percentile of up to 0.035 rad, at most 0.54 rad). The follow-the-gap agents were not affected.

The fix is on the `ope-benchmark` branch of `stochastic_ftg_agents`, which this repository uses. A quick re-run of two importance sampling estimators (seed 0) with and without the fix; the "old" numbers reproduce the results reported in the thesis, the metrics are computed as in `final_results.ipynb` (regret@1 in reward units):

TPDWIS (`run_iw.py --iw_type=step_wis_termination --ext=zero`)

| reward | Spearman (old -> fixed) | mean absolute error (old -> fixed) | regret@1 (old -> fixed) | max change of a pure pursuit estimate |
|---|---|---|---|---|
| progress | 0.81 -> 0.73 | 2.89 -> 3.40 | 0.00 -> 0.00 | 18% |
| checkpoint | 0.70 -> 0.70 | 4.76 -> 5.30 | 0.00 -> 0.00 | 22% |
| lifetime | 0.45 -> 0.51 | 7.23 -> 7.67 | 0.00 -> 0.00 | 16% |
| min_act | 0.66 -> 0.57 | 7.35 -> 8.20 | 16.95 -> 16.95 | 30% |

PDWIS (Mean) (`run_iw.py --iw_type=step_wis --ext=mean`)

| reward | Spearman (old -> fixed) | mean absolute error (old -> fixed) | regret@1 (old -> fixed) | max change of a pure pursuit estimate |
|---|---|---|---|---|
| progress | 0.67 -> 0.67 | 4.90 -> 5.17 | 0.00 -> 0.00 | 10% |
| checkpoint | 0.63 -> 0.57 | 6.43 -> 6.85 | 0.00 -> 0.00 | 10% |
| lifetime | 0.31 -> 0.17 | 11.31 -> 12.16 | 0.00 -> 0.95 | 11% |
| min_act | 0.59 -> 0.61 | 9.65 -> 10.24 | 16.95 -> 16.95 | 15% |

The last column is the largest change of a single pure pursuit estimate relative to its ground truth value; the follow-the-gap estimates are unchanged. With the fix the IS estimates get slightly worse: the mean absolute error increases by 0.3 to 0.9 in all cases, the rank correlation drops in 4 of the 8 cases, rises in 2 and is unchanged in 2, and the selected best policy only changes in one case. DR, FQE and the model-based methods are affected as well but have not been re-run.

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

Then execute:

```
bash dr_eval.bash
```


# Plotting and Visualization

Take a look at the notebook: `final_results.ipynb`
