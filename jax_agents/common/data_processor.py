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

"""Process data stream from interactions."""

from collections import deque
import jax.numpy as jnp
from jax import random


class ReplayBuffer():
    """A simple FIFO experience replay buffer for off-policy agents."""

    def __init__(self, buffer_size, state_dim, action_dim):
        """Initialize replay buffer with zeros."""
        self.states = jnp.zeros((buffer_size, state_dim))
        self.actions = jnp.zeros((buffer_size, action_dim))
        self.rewards = jnp.zeros((buffer_size))
        self.next_states = jnp.zeros((buffer_size, state_dim))
        self.ptr, self.size, self.buffer_size = 0, 0, buffer_size
        return

    def store(self, state, action, reward, next_state):
        """Store new experience."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.ptr = (self.ptr+1) % self.buffer_size
        self.size = min(self.size+1, self.buffer_size)
        return

    def sample_batch(self, batch_size):
        """Sample past experience."""
        if batch_size > self.size * 2:
            return None
        rng = random.PRNGKey(1996)
        indexes = random.randint(rng, shape=(batch_size),
                                 minval=0, maxval=self.size)
        data_batch = jnp.array(self.states[indexes],
                               self.actions[indexes],
                               self.rewards[indexes],
                               self.next_states[indexes])
        return jnp.transpose(data_batch)


class DataProcessor:
    """Class to process the data stream of states and actions.

    Calculate the rewards and store 3 tuples (state, action, reward) in a
    deque in order to support multistep reinforcement learning
    (see https://arxiv.org/pdf/1901.07510.pdf). Then fill the replay buffer
    for off policy rl algorithms.
    """

    def __init__(self, n_steps, buffer_size, state_dim, action_dim):
        """Initialize the multistep deque and the replay buffer."""
        self.n_steps_deque = deque(maxlen=n_steps+1)
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim)
        return

    def data_callback(self, state, action, reward_func, reset_flag):
        """Fill the deque and the replay buffer."""
        if len(self.n_steps_deque) == 0:
            self.n_steps_deque.append((state, action, 0.0))
            return
        reward = reward_func(self.n_steps_deque[-1][0],  # previous state
                             self.n_steps_deque[-1][1],  # previous action
                             state)
        cum_reward = self.n_steps_deque[-1][2] + reward
        self.n_steps_deque.append((state, action, cum_reward))
        if len(self.n_steps_deque) == self.n_steps_deque.maxlen:
            self.replay_buffer.store(self.n_steps_deque[0][0],  # first state
                                     self.n_steps_deque[0][1],  # first action
                                     cum_reward - self.n_steps_deque[0][2],
                                     state)
        if reset_flag:
            self.n_steps_deque.clear()
        return
