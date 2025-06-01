import numpy as np


def build_class_votation_matrix(labels, num_windows, num_samples, window_samples_step, window_samples_size):
  """
  The windows clustering step assign one cluster to each window, but finally
  we are not interested in the window's class but in the sample's class because
  at the end we need to know if a sample is a resting sample or an action sample.

  Because the windows can contains overlapping samples it's likely that one sample
  ends in more than one window with different label

  multichannel_signal: numpy array de ()
  """
  labels_count = []

  for window_idx in range(num_windows):
    window_label = labels[window_idx]
    channel_idx = 0
    t = np.full((num_samples, 1), np.inf)
    t[window_idx*window_samples_step: window_samples_size+(window_idx*window_samples_step)] = labels[window_idx]

    labels_count.append(t)

  arr = np.hstack(labels_count)
  sample_labels = []

  for i in arr:
    filtered_values = i[np.isfinite(i)]
    values, counts = np.unique(filtered_values, return_counts=True)
    most_common_label = int(values[counts.argmax()])
    sample_labels.append(most_common_label)

  # sample_labels contiene un elemento por sample, identificando asi la
  # categoria de cada sample
  return sample_labels