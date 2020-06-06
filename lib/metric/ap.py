import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist


def compute_precision(pairs):
  """Calculate precision"""

  return np.cumsum(pairs) / np.arange(1, len(pairs) + 1)


def compute_recall(pairs):
  """Calculate recall"""

  return np.cumsum(pairs) / np.sum(pairs)


def compute_ap(pairs):
  """Calculate average precision"""

  pos_indices = np.arange(1, np.sum(pairs) + 1)
  all_indices = np.arange(1, len(pairs) + 1)[pairs]

  return np.sum(pos_indices / all_indices) / np.sum(pairs)


def compute_prb(pairs):
  """Calculate precision-recall breakeven"""

  precision = compute_precision(pairs)

  # Multiple precisions can be at single recall point, take max
  for i in range(len(pairs) - 2, -1, -1):
    precision[i] = max(precision[i], precision[i + 1])

  recall = compute_recall(pairs)
  i = np.argmin(np.abs(recall - precision))

  return (recall[i] + precision[i]) / 2.


def single_view(x, y):
  """Calculate single view average precision."""

  n = len(x)

  dists = pdist(x, metric="cosine").astype(np.float32)
  del x

  indices = dists.argsort()
  del dists
  # print(y)
  pairs = np.zeros(n * (n - 1) // 2, dtype=np.bool)
  i = 0
  for j in range(n):
    pairs[i:(i + n - j - 1)][y[j] == y[j + 1:]] = True
    i += n - j - 1
  del y

  pairs = pairs[indices]
  del indices

  return compute_ap(pairs)


def cross_view(x1, y1, x2, y2):
  """Calculate crossview average precision."""

  assert len(y2) == len(set(y2.tolist()))

  n, m = len(x1), len(x2)

  dists = cdist(x1, x2, metric="cosine").astype(np.float32)
  del x1, x2

  indices = dists.ravel().argsort()
  del dists

  pairs = np.zeros((n, m), dtype=np.bool)
  for j in range(m):
    pairs[y1 == y2[j], j] = True
  del y1, y2

  pairs = pairs.ravel()[indices]
  del indices

  return compute_ap(pairs)
