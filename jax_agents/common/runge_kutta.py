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

"""Fourth order Runge-Kutta method for time-invariant ODEs."""


def runge_kutta(dynamics_f, state, action, dt):
    """dynamics_f(state, action) -> state_dot."""
    k1 = dynamics_f(state, action)
    k2 = dynamics_f(state + 0.5 * dt * k1, action)
    k3 = dynamics_f(state + 0.5 * dt * k2, action)
    k4 = dynamics_f(state + dt * k3, action)
    next_state = state + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    return next_state
