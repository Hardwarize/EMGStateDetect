from sklearn.cluster import KMeans


def detector_kmeans(features):
  """
  This function clusterize the windows in two clusters: action, rest
  """
  # Initialize KMeans with 2 clusters: 2 cluster because 2 states -> activity, rest
  kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
  # Fit and predict cluster labels
  window_labels = kmeans.fit_predict(features)
  return window_labels