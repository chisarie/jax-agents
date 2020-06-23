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

"""Collection of useful Neural Networks."""

import haiku as hk
import jax.numpy as jnp


# class FeedForwardPolicyNet():
#     """A simple feedforward neural network with tanh output."""

#     def __init__(self, output_sizes):
#         """Construct the MLP."""
#         self._net = hk.Sequential([hk.nets.MLP(output_sizes), jnp.tanh])
#         return

#     def __call__(self, x):
#         """Forward the network."""
#         return self._net

def mlp_policy_net(x, output_sizes):
    """Return simple feedforward neural network with tanh output."""
    net = hk.Sequential([hk.nets.MLP(output_sizes), jnp.tanh])
    return net(x)


def mlp_value_net(x, output_sizes):
    """Return simple feedforward neural network with tanh output."""
    net = hk.nets.MLP(output_sizes)
    return net(x)
