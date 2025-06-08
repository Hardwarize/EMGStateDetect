import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest


def detector_kmeans(features, remove_outliers: bool = True):
  """
  This function clusterize the windows in two clusters: action, rest
  """
  # Initialize KMeans with 2 clusters: 2 cluster because 2 states -> activity, rest
  kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
  # Fit and predict cluster labels
  if remove_outliers:
    iso = IsolationForest(contamination=0.05, random_state=42)
    labels = iso.fit_predict(features)

    features_replaced = features.copy()
    num_samples = features.shape[0]

    for i in range(num_samples):
        if labels[i] == -1:  # outlier
            # Search forward for the next inlier
            found = False
            for j in range(i + 1, num_samples):
                if labels[j] == 1:
                    features_replaced[i] = features[j]
                    found = True
                    break
            if not found:
                # If no inlier ahead, look backward
                for j in range(i - 1, -1, -1):
                    if labels[j] == 1:
                        features_replaced[i] = features[j]
                        found = True
                        break
  
    features = features_replaced.copy()
  window_labels = kmeans.fit_predict(features)
  return window_labels, kmeans.cluster_centers_ 


def fixed_time_70_pct(sampling_frequency: int, signal: np.ndarray):

  slices = []
  splits = [
    [0*sampling_frequency, 3*sampling_frequency], 
    [6*sampling_frequency, 9*sampling_frequency], 
    [12*sampling_frequency, 15*sampling_frequency]
  ]
  
  for idxs in splits:
    
    slice = signal[idxs[0]:idxs[1]]
    n = slice.shape[0]
    start = int(n * 0.15)
    end = int(n * 0.85)
    centered_70 = slice[start:end]
    slices.append(centered_70)

  clean_signal = np.concatenate(slices)
  return clean_signal