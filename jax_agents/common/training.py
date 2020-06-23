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

"""Simple training loop: interact with environment and do training step."""

from jax_agents.common.data_processor import DataProcessor, ReplayBuffer
from jax_agents.algorithms.random_agent import RandomAgent


def train(timesteps, environment, algorithm,
          n_steps, buffer_size, batch_size, seed):
    """Start training loop."""
    # Initialize data processor
    replay_buffer = ReplayBuffer(
        buffer_size, environment.state_dim, environment.action_dim, seed)
    data_processor = DataProcessor(n_steps, replay_buffer)
    # Initialize Networks
    state = environment.reset()
    random_agent = RandomAgent(seed, environment.action_dim)
    random_action = random_agent.select_action(state)
    algorithm.initialize_functions(state, random_action, seed)
    # Train loop
    for _ in range(timesteps):
        # Interact with environment
        normed_state = environment.norm_state(state)
        scaled_action = algorithm.select_action(normed_state)
        action = environment.rescale_action(scaled_action)
        reset_flag = environment.check_if_done(state)
        data_processor.data_callback(
            state, action, environment.reward_func, reset_flag)
        if reset_flag:
            state = environment.reset()
        else:
            state = environment.step(state, action)
        # Do training step
        if replay_buffer.size < batch_size * 2:
            continue
        data_batch = data_processor.replay_buffer.sample_batch(batch_size)
        algorithm.train_step(data_batch)
    return
