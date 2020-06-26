# MIT License

# Copyright (c) 2020 Authors:
#     - Eugenio Chisari <eugenio.chisari@gmail.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Example on how to use the Jax Agents API for training an agent."""

import datetime
import os
from jax_agents.common.training import train, TrainConfig
from jax_agents.common.simulation import simulate, render_csv
from jax_agents.environments.pendulum import PendulumEnv
from jax_agents.algorithms.ddpg import DDPG, DDPGConfig


def main():
    """Run the example: ddpg to solve the pendulum environment."""
    name = "ddpg_pendulum_example"
    time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder = os.getcwd() + "/examples/training/" + time + "/" + name+"/"
    env = PendulumEnv()
    ddpg_config = DDPGConfig(env.state_dim, env.action_dim)
    ddpg = DDPG(ddpg_config)
    train_config = TrainConfig(
        env=env, algorithm=ddpg, folder=folder, timesteps=int(2e4))
    train(train_config)
    simulate(environment=env, algorithm=ddpg, timesteps=200, folder=folder)
    render_csv(env, folder, timesteps=200)
    return


if __name__ == "__main__":
    main()
