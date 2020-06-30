# Jax Agents

<img src="docs/\_static//jax_agents_logo.png"
  align="right" width="30%"/>

[**Reference docs**](https://jax-agents.readthedocs.io/en/latest/)

Jax agents is a reinforcement learning library based on [google/jax](https://github.com/google/jax) and [deepmind/dm-haiku](https://github.com/deepmind/dm-haiku).

The goal and design philosophy of Jax Agents is to provide a simple to use library. To achieve this, all algorithms are self contained and use a consistent API.

NOTE: Jax Agents is currently under development, so expect the API to change.

## Installation

Jax Agents requires Python 3.7 or higher.
First install Jax and Haiku as described [here](https://dm-haiku.readthedocs.io/en/latest/index.html#installation).
You can then install Jax Agents using pip:

```bash
pip install git+https://github.com/chisarie/jax-agents
```

## Getting Started

Training a Reinforcement Learning agent in jax-agents is very simple:

```python
folder = "path/to/your/logging_folder/"
env = PendulumEnv()
ddpg_config = DDPGConfig(env.state_dim, env.action_dim)
algorithm = DDPG(ddpg_config)
train_timesteps = int(2e4)
train_config = TrainConfig(env, algorithm, folder, train_timesteps)
train(train_config)
plot_train_stats(folder)
sim_timesteps = 200
simulate(env, ddpg, sim_timesteps, folder)
render_csv(env, folder, sim_timesteps)
```

 This script will train an agent using the ddpg algorithm on the pendulum environment. During training, statistics are logged in the `monitor.csv` file. After training is complete, the statistics are plotted and a video
 of the trained agent is rendered. Complete and runnable scripts can be found in the `example` folder.

## Currently implemented algorithms

More algorithms will be added in the next weeks. Currently, the following algorithms are implemented:

* DDPG [[Paper](https://arxiv.org/abs/1509.02971)]

## How To Contribute

 Contributing guidelines will be released in the next weeks, once the project reaches a maturer stage.

## Citing Jax Agents

To cite this repository in your publications: 

```tex
@software{jax-agents2020github,
  author = {Eugenio Chisari},
  title = {Jax Agents: jax based reinforcement learning library},
  url = {https://github.com/chisarie/jax-agents},
  version = {0.0.1},
  year = {2020},
}
```

In the above bibtex entry, the version number is intended to be that from `setup.py`, and the year corresponds to the project's open-source release.
