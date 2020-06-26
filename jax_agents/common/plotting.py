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

"""Plotting utilities."""

import pandas as pd
import matplotlib.pyplot as plt


def plot_train_stats(path):
    """Plot training statistics."""
    stats = pd.read_csv(path+"monitor.csv")
    reward_mean = stats.reward.rolling(window=5).mean()
    reward_std = stats.reward.rolling(window=5).std()
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.plot(stats.timesteps, reward_mean)
    ax.fill_between(stats.timesteps, reward_mean-reward_std,
                    reward_mean+reward_std, alpha=0.4)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Reward')
    fig.savefig(path+"training_stats.png", bbox_inches='tight')
    return
