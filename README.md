# CreatureRL

By Jake Alt, Erik Luu, Nathan McCutchen, and Bharath Senthilkumar


This project took inspiration from a previous project and was rebooted for CSC 570 at Cal Poly, taught by Dr. Mugizi Robert Rwebangira

## External Resources

We used the following external resources in our project.

MuJoCo physics engine: https://mujoco.readthedocs.io/en/stable/overview.html  
Gymnasium RL framework: https://gymnasium.farama.org/  
Stable Baselines3 RL algorithms: https://stable-baselines3.readthedocs.io/en/master/index.html  

### Important Notice

At the time of writing this, there is a bug in the mujoco rendering file. If you encounter the following error, you need to navigate to `mujoco_rendering.py` in the gymnasium package on your local file system and replace the instance of `solver_iter` with `nsolver_iter` as the error suggests, and save your change.

This is the error:

```
AttributeError: 'mujoco._structs.MjData' object has no attribute 'solver_iter'. Did you mean: 'solver_niter'?
```

Since we use a virtual environment, the relative path to our `mujoco_rendering.py` file is `CreatureRL/.venv/lib/python3.10/site-packages/gymnasium/envs/mujoco/mujoco_rendering.py`.

## Setup

Follow these instructions to set up the codebase locally.

### 1. Clone the Repo

Run your favorite version of the git clone command on this repo. One version:

`git clone https://github.com/N8WM/CreatureRL.git`

### 2. Install Python

This code was developed and run on Python `3.10.10`, but most likely any version of Python `3.10` will do. Make sure you have an appropriate version installed locally.

### 3. Install Requirements

We recommend doing this in a fresh Python virtual environment. Cd into the repo and run:

`pip install -r requirements.txt`

## Quick Start

**Working Model**

As of 6/2/24, there are three successful models trained with the SAC algorithm in the `saved_models` directory: `pogodude_1`, `pogodude_2`, and `pogodude_2.1`. To run the most recent model, run the following command:

`python3 run.py -rsv 2.1`

**Other Options**

Our whole project has a single entry point, `run.py`. You can control the functionality via command-line arguments to Python script.

### Examples

Run an existing trained model (pogodude_\<VERSION\>.zip) and print evaluation data:

`python3 run.py -rsv <VERSION>`

Train a new model (pogodude_\<VERSION\>.zip) with 10,000,000 timesteps:

`python3 run_environment.py -tv <VERSION> -T 10000000`

```
usage: run.py [-h] (-t | -r) [-v VERSION] [-s] [-T TOTAL_TIMESTEPS] [-l LEARNING_RATE]

Run or train an agent to control a pogo robot

options:
  -h, --help            show this help message and exit

Functional arguments (mutually exclusive):
  -t, --train           train a new/existing model in test_models/
  -r, --run             run a model

Training and running arguments:
  -v VERSION, --version VERSION
                        version of the model to run (e.g. '1', '2', '2.1')

Running arguments:
  -s, --saved-dir       whether the model will be/is in the saved_models/ directory (otherwise test_models/)

Training arguments:
  -T TOTAL_TIMESTEPS, --total-timesteps TOTAL_TIMESTEPS
                        total number of timesteps to train the model for (default: 1,000,000)
  -l LEARNING_RATE, --learning-rate LEARNING_RATE
                        learning rate for training the model (default: 0.0003)
```
