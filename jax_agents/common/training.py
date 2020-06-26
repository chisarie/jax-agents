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
from dataclasses import dataclass
from typing import Any
from jax_agents.common.data_processor import DataProcessor, ReplayBuffer


@dataclass
class TrainConfig:
    """Config to initialize training loop.

    Args:
        env: environment to train on
        algorithm: rl algorithm to solve the problem
        folder: path to save the results
        timesteps: how long to train
        max_episode_len: when to reset the environment
        n_steps: support to multistep reinforcement learning
        buffer_size: how many transitions to store
        batch_size: used for training
        seed: random seed
    """

    env: Any
    algorithm: Any
    folder: str
    timesteps: int
    max_episode_len: int = 200
    n_steps: int = 1
    buffer_size: int = int(1e5)
    batch_size: int = 128
    seed: int = 1996

    def _initialize_data_processor(self):
        replay_buffer = ReplayBuffer(self.buffer_size, self.env.state_dim,
                                     self.env.action_dim, self.seed)
        self.data_processor = DataProcessor(
            self.n_steps, replay_buffer, self.folder)
        return

    def _sample_batch(self):
        return self.data_processor.replay_buffer.sample_batch(self.batch_size)


def train(config: TrainConfig):
    """Start the training loop with the given configuration."""
    config._initialize_data_processor()
    state = config.env.reset()
    reward = 0.0
    episode_len = 0
    for timestep in range(config.timesteps):
        # Interact
        state, reward, episode_len = _interaction_step(
            config, state, reward, timestep, episode_len)
        # Update
        config.algorithm.state = _update_step(config)
    config.data_processor.close()
    return


def _interaction_step(config, state, reward, timestep, episode_len):
    normed_state = config.env.norm_state(state)
    scaled_action = config.algorithm.select_action(
        normed_state, config.algorithm.func, config.algorithm.state)
    action = config.env.rescale_action(scaled_action)
    reset_flag = (config.env.check_if_done(state) or
                  episode_len == config.max_episode_len)
    config.data_processor.data_callback(
        normed_state, scaled_action, reward, reset_flag, timestep)
    if reset_flag:
        return config.env.reset(), 0.0, 0
    next_state = config.env.step(state, action)
    reward = config.env.reward_func(state, action, next_state)
    return next_state, reward, episode_len + 1


def _update_step(config):
    if config.data_processor.replay_buffer.size < config.batch_size * 2:
        return config.algorithm.state
    data_batch = config._sample_batch()
    return config.algorithm.train_step(
        data_batch, config.algorithm.func, config.algorithm.state)
