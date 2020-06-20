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

from collections import deque


class DataProcessor:
    """Class to process the data stream of states and actions.

    It calculates the rewards and stores 3 tuples (state, action, reward) in a
    deque in order to support multistep reinforcement learning
    (see https://arxiv.org/pdf/1901.07510.pdf).

    :param reward_func: (func) reward_func(state, action) -> reward
    :param n_steps: (int) how many steps to consider for the multistep setting,
        use 1 for standard setting
    """

    def __init__(self, reward_func, n_steps):
        """Initialize reward function and the multistep deque."""
        self.reward_func = reward_func
        self.n_steps_deque = deque(maxlen=n_steps+1)
        return
