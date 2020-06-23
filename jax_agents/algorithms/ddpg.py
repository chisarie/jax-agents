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

from collections import namedtuple
import jax
import jax.numpy as jnp
from jax.experimental import optix


DDPG = namedtuple("DDPG", [
                  "pi_net",
                  "q_net",
                  "pi_optimizer",
                  "q_optimizer",
                  "gamma",
                  "state_dim",
                  "action_dim"])


DDPGState = namedtuple("DDPGState", [
                       "pi_params",
                       "q_params",
                       "pi_opt_state",
                       "q_opt_state"])


def initialize_algo_state(state, action, seed, algo):
    """Initialize pi, q parameters and optimizers states."""
    rng = jax.random.PRNGKey(seed)  # rundom number generator
    state_action = jnp.hstack((state, action))
    pi_params = algo.pi_net.init(rng, state)
    q_params = algo.q_net.init(rng, state_action)
    pi_opt_state = algo.pi_optimizer.init(pi_params)
    q_opt_state = algo.q_optimizer.init(q_params)
    return DDPGState(pi_params, q_params, pi_opt_state, q_opt_state)


def select_action(state, algo, algo_state):
    """Output on policy action."""
    return algo.pi_net.apply(algo_state.pi_params, state)


def train_step(data_batch, algo, algo_state):
    """Update all functions."""
    # Update Pi
    pi_grads = jax.grad(_pi_loss_batched)(algo_state.pi_params, data_batch)
    pi_updates, new_pi_opt_state = algo.pi_optimizer.update(
        pi_grads, algo_state.pi_opt_state)
    new_pi_params = optix.apply_updates(algo_state.pi_params, pi_updates)
    # Update Q
    q_grads = jax.grad(_q_loss_batched)(algo_state.q_params, data_batch)
    q_updates, new_q_opt_state = algo.q_optimizer.update(
        q_grads, algo_state.q_opt_state)
    new_q_params = optix.apply_updates(algo_state.q_params, q_updates)
    return DDPGState(
        new_pi_params, new_q_params, new_pi_opt_state, new_q_opt_state)


def _pi_loss(pi_params, data_point, algo, algo_state):
    state = data_point[:algo.state_dim]
    pi_state = algo.pi_net.apply(pi_params, state)
    q_state_pi_state = algo.q_net.apply(
        algo_state.q_params, jnp.hstack((state, pi_state)))
    loss = - q_state_pi_state
    return loss


def _pi_loss_batched(self, pi_params, batched_data, algo, algo_state):
    return jnp.mean(jax.vmap(_pi_loss, in_axes=(None, 0, None, None))(
        pi_params, batched_data, algo, algo_state))


def _q_loss(self, q_params, data_point, algo, algo_state):
    state, action, reward, next_state = _expand_data_tuple(data_point, algo)
    pi_next_state = algo.pi_net.apply(algo_state.pi_params, next_state)
    target_value = algo.q_net.apply(
        q_params, jnp.hstack((next_state, pi_next_state)))
    bellman_target = reward + algo.gamma * target_value
    q_state_action = algo.q_net.apply(
        q_params, jnp.hstack((state, action)))
    loss = jnp.square(bellman_target - q_state_action)
    return loss


def _q_loss_batched(self, q_params, batched_data, algo, algo_state):
    return jnp.mean(jax.vmap(_q_loss, in_axes=(None, 0, None, None))(
        q_params, batched_data, algo, algo_state))


def _expand_data_tuple(data_point, algo):
    state = data_point[:algo.state_dim]
    action = data_point[algo.state_dim:algo.state_dim+algo.action_dim]
    reward = data_point[algo.state_dim+algo.action_dim]
    next_state = data_point[algo.state_dim+algo.action_dim+1:]
    return state, action, reward, next_state
