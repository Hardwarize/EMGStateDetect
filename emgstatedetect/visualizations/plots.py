import itertools

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


def plot_multisubject_multiclass_multichannel(signal: np.ndarray, fig_width: int = 30, fig_height: int = 10) -> Figure:
    """
    signal: shape (subjects, classes, channels, samples)
    """
    fig = plt.figure(figsize=(fig_width, fig_height))
    subject_dim, classes_dim, signal_dim, num_samples = signal.shape
    outer_gs = fig.add_gridspec(subject_dim, classes_dim, hspace=0.1)

    for idx in list(itertools.product(list(range(subject_dim)), list(range(classes_dim)))):
        raw0_gs = outer_gs[idx[0], idx[1]].subgridspec(signal_dim, 1, hspace=0)
        for i in range(signal_dim):
            ax = fig.add_subplot(raw0_gs[i, 0])
            ax.plot(signal[idx[0], idx[1], i])

    return fig
