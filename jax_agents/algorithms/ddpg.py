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

from functools import partial
from typing import Callable, Any, NamedTuple
import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental import optix

from jax_agents.common.networks import mlp_policy_net, mlp_value_net


class DDPGFunc(NamedTuple):
    """Config to initialize the DDPG functions.

    Args:
        pi_net: policy neural network
        q_net: q function neural network
        pi_optimizer: policy optimizer (adam)
        q_optimizer: q function optimizer (adam)
        gamma: discount factor
        state_dim: dimension of the state vector.
        action_dim: dimension of the action vector.
    """

    pi_net: Callable
    q_net: Callable
    pi_optimizer: Callable
    q_optimizer: Callable
    gamma: float
    state_dim: int
    action_dim: int


class DDPGState(NamedTuple):
    """State of the DDPG networks.

    Args:
        pi_params: policy neural network parameters
        q_params: q function neural network parameters
        pi_opt_state: state of the policy optimizer (adam)
        q_opt_state: state of the q function optimizer (adam)
    """

    pi_params: Any
    q_params: Any
    pi_opt_state: Any
    q_opt_state: Any


class DDPGConfig(NamedTuple):
    """Config to initialize DDPG.

    Args:
        state_dim: the dimension of the state vector.
        action_dim: the dimension of the action vector.
        pi_net_size: a list of int corresponding to the hidden sizes of the
            policy network.
        q_net_size: a list of int corresponding to the hidden sizes of the
            q network.
        learning_rate: the learning rate of the adam optimizer.
        gamma: the discount factor of the algorithm.
        seed: the random seed for initialization of the networks.
    """

    state_dim: int
    action_dim: int
    pi_net_size: tuple = (64, 64)
    q_net_size: tuple = (64, 64)
    learning_rate: float = 1e-3
    gamma: float = 0.99
    seed: int = 1996


class DDPG():
    """DDPG Algorithm class. "https://arxiv.org/abs/1509.02971"."""

    def __init__(self, config: DDPGConfig):
        """Initialize the algorithm."""
        pi_net = partial(mlp_policy_net,
                         output_sizes=config.pi_net_size+(config.action_dim,))
        pi_net = hk.transform(pi_net)
        q_net = partial(mlp_value_net,
                        output_sizes=config.q_net_size + (1,))
        q_net = hk.transform(q_net)
        pi_optimizer = optix.adam(learning_rate=config.learning_rate)
        q_optimizer = optix.adam(learning_rate=config.learning_rate)
        self.func = DDPGFunc(pi_net, q_net, pi_optimizer, q_optimizer,
                             config.gamma, config.state_dim, config.action_dim)
        rng = jax.random.PRNGKey(config.seed)  # rundom number generator
        state = jnp.zeros(config.state_dim)
        state_action = jnp.zeros(config.state_dim + config.action_dim)
        pi_params = pi_net.init(rng, state)
        q_params = q_net.init(rng, state_action)
        pi_opt_state = pi_optimizer.init(pi_params)
        q_opt_state = q_optimizer.init(q_params)
        self.state = DDPGState(pi_params, q_params, pi_opt_state, q_opt_state)
        return

    @partial(jax.jit, static_argnums=(0, 2))
    def select_action(self, state, algo_func, algo_state):
        """Output on policy action."""
        return algo_func.pi_net.apply(algo_state.pi_params, state)

    @partial(jax.jit, static_argnums=(0, 2))
    def train_step(self, data_batch, algo_func, algo_state):
        """Update all functions."""
        # Update Pi
        pi_grads = jax.grad(_pi_loss_batched)(algo_state.pi_params, data_batch,
                                              algo_func, algo_state)
        pi_updates, new_pi_opt_state = algo_func.pi_optimizer.update(
            pi_grads, algo_state.pi_opt_state)
        new_pi_params = optix.apply_updates(algo_state.pi_params, pi_updates)
        # Update Q
        q_grads = jax.grad(_q_loss_batched)(algo_state.q_params, data_batch,
                                            algo_func, algo_state)
        q_updates, new_q_opt_state = algo_func.q_optimizer.update(
            q_grads, algo_state.q_opt_state)
        new_q_params = optix.apply_updates(algo_state.q_params, q_updates)
        return DDPGState(
            new_pi_params, new_q_params, new_pi_opt_state, new_q_opt_state)


def _pi_loss(pi_params, data_point, algo_func, algo_state):
    state = data_point[:algo_func.state_dim]
    pi_state = algo_func.pi_net.apply(pi_params, state)
    q_state_pi_state = algo_func.q_net.apply(
        algo_state.q_params, jnp.hstack((state, pi_state)))
    loss = - q_state_pi_state
    return loss


def _pi_loss_batched(pi_params, batched_data, algo_func, algo_state):
    return jnp.mean(jax.vmap(_pi_loss, in_axes=(None, 0, None, None))(
        pi_params, batched_data, algo_func, algo_state))


def _q_loss(q_params, data_point, algo_func, algo_state):
    state, action, reward, next_state = _expand_data_tuple(
        data_point, algo_func.state_dim, algo_func.action_dim)
    pi_next_state = algo_func.pi_net.apply(algo_state.pi_params, next_state)
    target_value = algo_func.q_net.apply(
        q_params, jnp.hstack((next_state, pi_next_state)))
    bellman_target = reward + algo_func.gamma * target_value
    q_state_action = algo_func.q_net.apply(
        q_params, jnp.hstack((state, action)))
    loss = jnp.square(bellman_target - q_state_action)
    return loss


def _q_loss_batched(q_params, batched_data, algo_func, algo_state):
    return jnp.mean(jax.vmap(_q_loss, in_axes=(None, 0, None, None))(
        q_params, batched_data, algo_func, algo_state))


def _expand_data_tuple(data_point, state_dim, action_dim):
    state = data_point[:state_dim]
    action = data_point[state_dim:state_dim+action_dim]
    reward = data_point[state_dim+action_dim]
    next_state = data_point[state_dim+action_dim+1:]
    return state, action, reward, next_state
