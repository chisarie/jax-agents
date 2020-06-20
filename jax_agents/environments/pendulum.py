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

"""A simple inverted pendulum environment."""

import numpy as np
from jax_agents.common.runge_kutta import runge_kutta


def pendulum_dynamics(state, action):
    """Environment's dynamics.

    Args:
        state (vector):
            state1 : angle (theta)
            state2 : angular velocity (theta_dot)
        action (scalar): torque
    Returns:
        x_dot (vector): derivative of the state
    """
    # Constants:
    g = 9.8         # acceleration due to gravity, positive downward, [m/sec^2]
    m = 0.5         # mass at the tip of the pendulum [kg]
    length = 1.0    # pole length [meters]
    b = 0.01        # friction coefficient [kg*m^2/s]

    # States
    theta = state[0]
    theta_dot = state[1]

    x_dot = np.empty(2)
    x_dot[0] = theta_dot
    x_dot[1] = (action - np.sin(theta) * g / length
                - theta_dot * b / (m * length**2))
    return x_dot


class PendulumEnv():
    """
    The pendulum starts almost upright, the goal is to prevent falling down.

    States:
        Type: Box(4)
        Num	State                       Min         Max
        0	Pendulum Angle              0           2 pi
        2	Pendulum Angular Velocity   -Inf        Inf

    Actions:
        Type: Box(1)
        Num	Action                      Min         Max
        0	Torque                      -4.0        4.0

    Reward:
        Reward is 1 for every step in the upright position, -0.1 otherwise
    Starting State:
        All states equal to zero except pole angle = pi + (-10, +10)
        (i.e. pointing upwards)
    Episode Length:
        10 seconds, i.e. 200 steps at 20 Hz
    """

    def step(self, state, action):
        """Integrate the dynamics on step forward."""
        dt = 0.05
        next_state = runge_kutta(pendulum_dynamics, state, action, dt)
        return next_state
