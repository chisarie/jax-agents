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


def simulate(environment, policy, timesteps, folder):
    """Simulate one episode."""
    state = environment.reset()
    field_names = environment.state_names + environment.action_names
    field_names += ["reward", "reset_flag"]
    sim_logger = SimLogger(folder, field_names)
    for _ in range(timesteps):
        normed_state = environment.norm_state(state)
        scaled_action = policy.select_action(normed_state)
        action = environment.rescale_action(scaled_action)
        next_state = environment.step(state, action)
        reward = environment.reward_func(state, action, next_state)
        reset_flag = environment.check_if_done(next_state)
        sim_logger.log(state + action + [reward, reset_flag])
        if reset_flag:
            state = environment.reset()
        else:
            state = next_state
    sim_logger.close()
    return


class SimLogger():
    """Log data from simulation in a csv file."""

    def __init__(self, folder, field_names):
        """Initialize csv file."""
        # Create csv
        self.csv_file = open(folder+"sim.csv", 'w', newline='')
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
