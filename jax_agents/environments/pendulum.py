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

from jax import random
import jax.numpy as jnp
from jax.ops import index_update
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from jax_agents.common.runge_kutta import runge_kutta


class PendulumEnv():
    """
    The pendulum starts almost upright, the goal is to prevent falling down.

    States:
        Type: Box(4)
        Num	State                       Min         Max
        0	Pendulum Angle              0(downward) 2 pi
        2	Pendulum Angular Velocity   -Inf        Inf
    Actions:
        Type: Box(1)
        Num	Action                      Min         Max
        0	Torque                      -4.0        4.0
    Orientation:
        angle: pi/2 -> to the right
        angular velocity: positive -> counterclockwise
    Reward:
        Reward is 1 for every step in the upright position, -0.1 otherwise
    Starting State:
        All states equal to zero except pole angle = pi + (-10, +10)
        (i.e. pointing upwards)
    Episode Length:
        10 seconds, i.e. 200 steps at 20 Hz
    """

    def __init__(self, seed):
        """Initialize environment."""
        self.rng = random.PRNGKey(seed)  # rundom number generator
        self.length = 1.0                # pole length [meters]
        self.dt = 0.05                   # timesteps length [seconds]
        self.state_dim = 2               # dimension of state vector
        self.action_dim = 1              # dimension of action vector
        self.state_names = ["theta", "theta_dot"]
        self.action_names = ["torque"]
        return

    def norm_state(self, state):
        """Norm the state to have zero mean and unit variance."""
        state_mean = jnp.array([jnp.pi, 0.0])
        state_std = jnp.array([1.8, 2.0])
        normed_state = (state - state_mean) / state_std
        return normed_state

    def rescale_action(self, scaled_action):
        """Rescale the action from the policy (which are between -1 and 1)."""
        return scaled_action * 4.0

    def step(self, state, action):
        """Integrate the dynamics on step forward."""
        next_state = runge_kutta(self.pendulum_dynamics, state,
                                 action, self.dt)
        # wrap angle in [0, 2pi]
        next_state = index_update(next_state, 0, next_state[0] % (2*jnp.pi))
        return next_state

    def reward_func(self, state, action, next_state):
        """Return the reward for the given transition."""
        distance_to_upright = abs(next_state[0] - jnp.pi)
        if distance_to_upright < jnp.deg2rad(10):
            reward = 1.0
        else:
            reward = -0.1
        return reward

    def reset(self):
        """Reset the state to start a new episode."""
        self.rng, rng_input = random.split(self.rng)
        angle = jnp.pi + random.uniform(
            rng_input, minval=-jnp.deg2rad(10), maxval=jnp.deg2rad(10))
        angular_velocity = 0.0
        return jnp.array([angle, angular_velocity])  # state

    def check_if_done(self, state):
        """Check if episode needs to restart."""
        return False

    def pendulum_dynamics(self, state, action):
        """Environment's dynamics. Return the derivative of the state."""
        # Constants:
        g = 9.8         # gravity, positive downward, [m/sec^2]
        m = 0.5         # mass at the tip of the pendulum [kg]
        b = 0.01        # friction coefficient [kg*m^2/s]

        # States
        theta = state[0]
        theta_dot = state[1]
        torque = action[0]

        x_dot = jnp.array([theta_dot,
                          (torque - jnp.sin(theta) * g / self.length
                           - theta_dot * b / (m * self.length**2))
                           ])
        return x_dot

    def initialize_rendering(self, log, fig):
        """Initialize Matplotlib axes for rendering."""
        ax = fig.add_subplot()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        angle = float(log[0])
        x = self.length * jnp.sin(angle)
        y = -self.length * jnp.cos(angle)
        self.line = mlines.Line2D([0.0, x], [0.0, y],
                                  linewidth=2, color="tab:brown")
        self.circle = mpatches.Circle((x, y), radius=0.1, color="black")
        ax.add_line(self.line)
        ax.add_patch(self.circle)
        return [self.line, self.circle]

    def render(self, log):
        """Render the given timestep."""
        angle = float(log[0])
        x = self.length * jnp.sin(angle)
        y = -self.length * jnp.cos(angle)
        self.line.set_xdata([0.0, x])
        self.line.set_ydata([0.0, y])
        self.circle.center = (x, y)
        return [self.line, self.circle]
