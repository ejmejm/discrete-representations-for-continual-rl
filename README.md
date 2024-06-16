# Harnessing Discrete Representations for Continual Reinforcement Learning

This directory contains both code and data used to produce the work for the paper "[Harnessing Discrete Representations for Continual Reinforcement Learning](https://arxiv.org/abs/2312.01203)".

**Paper link:** [https://arxiv.org/abs/2312.01203](https://arxiv.org/abs/2312.01203)

**Paper authors:** Edan Meyer, Adam White, Marlos Machado

**Code author:** Edan Meyer

## Code Structure

The general structure of the directories is as follows:
  - `pip_env.txt`: A dump of the `pip freeze` command of the python environment used to run results. Many of the listed libraries are not required, but this file can be used to determine which version of a given module was used.
  - `shared/`: Code for models and training shared between many experiments.
  - `discrete_mbrl/`:
    - `full_train_eval.py`: Entry point for most model learning experiments.
    - `model_construction.py`: Handles the building of different architectures and matching them with appropriate training classes.
    - `training_helpers.py`: Specifies arguments for most experiments, and contains some shared training code.
    - `model_free/`: Code specific to training RL models.
    - `sweep_configs/`: Experiment runs and sweeps are managed with either Weights & Biases or CometML depending on the command line arguments used. This directory contains the configs for running all of the experiments in the paper (any maybe some more).
    - `analysis/`: Analysis code and results. Each experiment is separated into its own notebook. Data has been included so that figures can be reproduced.
