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

"""Deep Deterministic Policy Gradient (DDPG)."""

import haiku as hk
import jax


class FeedForwardNetwork():
    """A simple feedforward neural network with tanh output."""

    def __init__(self, output_sizes):
        """Construct the MLP."""
        self._net = hk.Sequential([hk.nets.MLP(output_sizes), jax.lax.tanh])
        return

    def __call__(self, x):
        """Forward the network."""
        return self._net


class DDPG():
    """Algorithm class."""

    def __init__(self, pi_net, q_net, pi_optimizer, q_optimizer):
        """Initialize pi, q functions, and optimizers."""
        self.pi_net = hk.transform(pi_net)
        self.q_net = hk.transform(q_net)
        self.pi_optimizer = pi_optimizer
        self.q_optimizer = q_optimizer
        return

    def initialize_functions(self, data_point):
        """Initialize pi, q parameters and optimizers states."""
        state = data_point[0]
        action = data_point[1]
        state_action = jax.numpy.hstack((state, action))
        self.pi_params = self.pi_net.init(jax.random.PRNGKey(42), state)
        self.q_params = self.q_net.init(jax.random.PRNGKey(27), state_action)
        self.pi_opt_state = self.pi_optimizer.init(self.pi_params)
        self.q_opt_state = self.q_optimizer.init(self.q_params)
        return

    def train_step(self, data_batch):
        """Update all functions."""
        pi_grads = jax.grad(self._pi_loss)(self.pi_params, data_batch)
        # TODO
        return

    def _pi_loss(self, data_point):
        state, action, reward, next_state = data_point
        loss = 0.0 # TODO
        return loss

    def _pi_loss_batched(self, batched_data):
        return jax.vmap(self._pi_loss)(batched_data)

    def _q_loss(self, data_point):
        state, action, reward, next_state = data_point
        loss = 0.0 # TODO
        return loss

    def _q_loss_batched(self, batched_data):
        return jax.vmap(self._q_loss)(batched_data)
