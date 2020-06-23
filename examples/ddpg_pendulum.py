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
from functools import partial
from jax.experimental import optix
import haiku as hk

from jax_agents.common.training import train
from jax_agents.environments.pendulum import PendulumEnv
from jax_agents.algorithms.ddpg import DDPG
from jax_agents.common.networks import mlp_policy_net, mlp_value_net


def main():
    """Run the example."""
    random_seed = 1996
    name = "ddpg_pendulum_example"
    time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder = os.getcwd() + "/examples/training/" + time + "/" + name+"/"
    timesteps = int(1e4)
    environment = PendulumEnv(random_seed)

    # Put this in separate function with args: output_sizes, learning_rate. And out: algo
    pi_net = hk.transform(partial(
        mlp_policy_net, output_sizes=[64, 64, environment.action_dim]))
    q_net = hk.transform(partial(mlp_value_net, output_sizes=[64, 64, 1]))
    pi_optimizer = optix.adam(learning_rate=1e-3)
    q_optimizer = optix.adam(learning_rate=1e-3)
    algorithm = DDPG(pi_net, q_net, pi_optimizer, q_optimizer)

    episode_len = 120
    n_steps = 1
    buffer_size = int(1e4)
    batch_size = 128
    train(timesteps, environment, algorithm, episode_len, n_steps,
          buffer_size, batch_size, random_seed, folder)
    return


if __name__ == "__main__":
    main()
