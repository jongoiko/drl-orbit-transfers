Training code for the work titled _Adaptive guidance and control for spacecraft orbit transfers using deep reinforcement learning_, presented at the [XXII Workshop on Celestial Mechanics](https://jtmc2024.udg.edu/?page_id=89&lang=en) in Girona, Spain.

<p align="center">
  <img width="300px" src="https://github.com/jongoiko/orbit/blob/main/orbit.png">
</p>

## Installation and running

After cloning the repo, create an environment with the dependencies (using e.g. [conda](https://conda.io/projects/conda/en/latest/index.html)):

```
$ cd drl-orbit-transfers
$ conda env create --file environment.yml
$ conda activate drl-orbit-transfers
```

Create a directory for logging with [TensorBoard](https://www.tensorflow.org/tensorboard?hl=es-419), and another for logging trajectories.
Then, to start a training run for an experiment, e.g. `src/experiments/ecc_change.csv`:

```
$ mkdir tensorboard trajectories
$ python src/train_ppo.py src/experiments/ecc_change.csv --tb_log_dir tensorboard --traj_log_dir trajectories
```

To visualize a training trajectory (we also need to provide the experiment file):

```
$ python src/plot_trajectory.py trajectories/[trajectory file] src/experiments/ecc_change.csv
```
