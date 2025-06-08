import itertools

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection


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


def create_line_collection(values_array, color_array, ax, color_map):
    points = np.array([list(range(values_array.shape[0])), values_array]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    line_collections = []
    for color_value in np.unique(color_array):
        mask = color_array[:-1] == color_value
        seg = segments[mask]
        lc = LineCollection(seg, color=color_map[color_value], label=color_value)
        ax.add_collection(lc)
        line_collections.append(lc)

    return line_collections


def plot_raw_segmented_clean_one_subject_one_class(raw_signal, segmented_signal, clean_signal, classes_dict, labels) -> Figure:
    
    num_channels = 4
    
    fig = plt.figure(figsize=(24, 8))
    # 3 -> raw, segmented, clean
    gs0 = fig.add_gridspec(1, 3)
    raw_gs = gs0[0].subgridspec(num_channels, 1, hspace=0)
    segmented_gs = gs0[1].subgridspec(num_channels, 1, hspace=0)
    clean_gs = gs0[2].subgridspec(num_channels, 1, hspace=0)
    
    for channel_idx in range(num_channels):
        ax_raw = fig.add_subplot(raw_gs[channel_idx])
        ax_segmented = fig.add_subplot(segmented_gs[channel_idx])
        ax_clean = fig.add_subplot(clean_gs[channel_idx])
        
        ax_raw.plot(raw_signal[:,channel_idx])
        #ax_segmented.plot(segmented_signal[:,channel_idx])
        ax_clean.plot(clean_signal[:,channel_idx])
        
                
        values = segmented_signal[:, channel_idx]
        color_map = {
            classes_dict["Action"]["label"]: 'blue',
            classes_dict["Rest"]["label"]: 'red'
        }
    
        legend_labels = {
            classes_dict["Action"]["label"]: 'Action',
            classes_dict["Rest"]["label"]: 'Rest',
        }
        
        
        line = create_line_collection(values, labels, ax_segmented, color_map)
        ax_segmented.set_ylim(values.min(), values.max())
        ax_segmented.set_xlim(0, values.shape[0])
    
    return fig
    