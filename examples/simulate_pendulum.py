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

"""Example on how to use the Jax Agents API for running a simulation."""
import datetime
import os
from functools import partial
from jax import random

from jax_agents.common.simulation import simulate, render_csv
from jax_agents.algorithms.random_agent import RandomAgent
from jax_agents.environments import pendulum


def main():
    """Run the example."""
    random_seed = 56
    sim_name = "random_pendulum_example"
    time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder = os.getcwd() + "/examples/simulations/" + time + "/" + sim_name+"/"
    environment = pendulum.PendulumEnv(random_seed)
    policy = RandomAgent(random_seed, environment.action_dim)
    timesteps = 200
    simulate(environment, policy, timesteps, folder)
    render_csv(environment, folder, timesteps)
    return


if __name__ == "__main__":
    main()
