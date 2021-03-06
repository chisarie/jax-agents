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

"""Simulate a policy in the given environment, optionally render a video."""

import csv
import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def simulate(environment, algorithm, timesteps, folder):
    """Simulate one episode."""
    state = environment.reset()
    field_names = environment.state_names + environment.action_names
    field_names += ["reward", "reset_flag"]
    sim_logger = SimLogger(folder, field_names)
    for _ in range(timesteps):
        normed_state = environment.norm_state(state)
        scaled_action = algorithm.select_action(
            normed_state, algorithm.func, algorithm.state)
        action = environment.rescale_action(scaled_action)
        next_state = environment.step(state, action)
        reward = environment.reward_func(state, action, next_state)
        reset_flag = environment.check_if_done(next_state)
        sim_logger.log(jnp.hstack((state, action, reward, reset_flag)))
        if reset_flag:
            state = environment.reset()
        else:
            state = next_state
    sim_logger.close()
    return


def render_csv(env, folder, timesteps):
    """Load the csv file from the simulation and render it."""
    fig = plt.figure(dpi=140, tight_layout=True)
    csv_reader_gen = (row.split(',') for row in open(folder + "sim.csv"))
    next(csv_reader_gen)  # ignore first line (header)
    env.initialize_rendering(next(csv_reader_gen), fig)
    animation = FuncAnimation(fig,
                              func=env.render,
                              frames=csv_reader_gen,
                              blit=True,
                              save_count=timesteps-5,
                              repeat=False,
                              interval=env.dt * 1000)  # milliseconds
    animation.save(folder + "animation.mp4")
    return


class SimLogger():
    """Log data from simulation in a csv file."""

    def __init__(self, folder, field_names):
        """Initialize csv file."""
        os.makedirs(folder, exist_ok=True)
        self.csv_file = open(folder + "sim.csv", 'w', newline='')
        self.field_names = field_names
        self.writer = csv.DictWriter(self.csv_file, fieldnames=field_names)
        self.writer.writeheader()
        return

    def log(self, log_list):
        """Append data in file."""
        log_data_dict = dict(zip(self.field_names, log_list))
        self.writer.writerow(log_data_dict)
        return

    def close(self):
        """Close file."""
        self.csv_file.close()
        return
