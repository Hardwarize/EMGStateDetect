import numpy as np
from scipy import stats


def map_window_labels_to_samples_by_voting(labels, window_samples_size, window_samples_step, total_samples):
    """
    Assigns a label to each sample by mapping window-based cluster labels (from k-means) 
    to individual samples using majority voting.

    Each sample may belong to multiple overlapping windows. For each sample, the final 
    label is determined by the most frequent window label among all windows that include the sample.

    Args:
        window_labels (array-like): Array of cluster labels assigned to each window.
        window_indices (list of array-like): List where each element contains the indices 
            of the samples that belong to the corresponding window.
        n_samples (int): Total number of samples in the signal.

    Returns:
        sample_labels (np.ndarray): Array of shape (n_samples,) with the label assigned 
            to each sample based on majority voting.
    """
    
    assert isinstance(labels, np.ndarray) and labels.ndim == 1, "Variable must be a 1D numpy array."
    
    non_window_elements = total_samples - window_samples_size
    
    samples_label_voting_matrix = np.vstack(
        [
        np.tile(labels, (window_samples_size, 1)),
        np.full((non_window_elements, labels.shape[0]), np.nan)
        ]
    )
    
    for i in range(samples_label_voting_matrix.shape[1]):
        samples_label_voting_matrix[:, i] = np.roll(samples_label_voting_matrix[:, i], i*window_samples_step)
        
    sample_labels = stats.mode(samples_label_voting_matrix, axis=1, nan_policy='omit').mode
    
    return sample_labels


def assign_class_to_labels(multichannel_signal, labels_array):
    """
    This function takes a multichannel signal and an array of labels,
    and an array of labels, then try to check which label corresponds to the action and which one to the rest.
    It returns a dictionary with two keys: "Rest" and "Action", each containing the label and the corresponding data.
    """

    unique_labels = np.unique(labels_array).astype(np.int64) 
    assert unique_labels.shape[0] == 2, "There must be exactly two unique labels in the labels array. One for action and one for rest."

    # Get signals segments by labels
    split_arrays = [(label, multichannel_signal[labels_array == label,:]) for label in unique_labels]
    
    # Get the mean absolute value of each segment
    # Hopefully the segment with the highest mean absolute value corresponds to the action
    # and the segment with the lowest mean absolute value corresponds to the rest.
    info = [(split_array[0], split_array[1], np.abs(split_array[1]).mean()) for split_array in split_arrays]

    if info[0][2] > info[1][2]:
        return {
        "Rest" : {"label": info[1][0], "data": info[1][1]},
        "Action" : {"label": info[0][0], "data": info[0][1]}
        }
    else:
        return {
        "Rest"  : {"label": info[0][0], "data": info[0][1]},
        "Action"  : {"state": info[1][0], "data": info[1][1]}
        }
