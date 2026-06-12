import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors


def run_kmeans(matrix, k):
    """
    Partition pixels into k clusters.
    Returns integer label array of shape (n_pixels,).
    """
    n_init = 3 if matrix.shape[0] > 1_000_000 else 10
    km = KMeans(n_clusters=int(k), algorithm="elkan", n_init=n_init, random_state=0)
    return km.fit_predict(matrix)


def run_dbscan(matrix, eps, min_samples):
    """
    Density-based clustering. Noise pixels get label -1.
    Returns integer label array of shape (n_pixels,).
    """
    db = DBSCAN(eps=float(eps), min_samples=int(min_samples))
    return db.fit_predict(matrix)


def run_isolation_forest(matrix, contamination):
    """
    Anomaly detection. Returns (scores, labels) where labels are 1 (normal)
    or -1 (anomaly), and scores are the raw decision function values
    (more negative = more anomalous).
    """
    n_estimators = 50 if matrix.shape[0] > 1_000_000 else 100
    clf = IsolationForest(n_estimators=n_estimators, contamination=float(contamination), random_state=0)
    labels = clf.fit_predict(matrix)
    scores = clf.decision_function(matrix)
    return scores, labels


def compute_knn_distances(matrix, k):
    """
    Compute the distance from each point to its k-th nearest neighbour.
    Returns a 1-D array sorted in ascending order (the k-NN distance plot).
    """
    k = max(1, int(k))
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(matrix)
    distances, _ = nn.kneighbors(matrix)
    kth_distances = distances[:, k]
    return np.sort(kth_distances)


def detect_elbow(distances):
    """
    Locate the elbow in a sorted k-NN distance curve using the perpendicular
    distance method: find the point furthest from the line connecting the
    first and last points of the normalised curve. This targets the lift-off
    point rather than the top of the steep rise.
    """
    if len(distances) < 3:
        return float(distances[-1])

    x = np.linspace(0.0, 1.0, len(distances))
    y = (distances - distances.min()) / (distances.max() - distances.min() + 1e-12)

    # Perpendicular distance from each point to the line (x[0], y[0]) → (x[-1], y[-1])
    x1, y1, x2, y2 = x[0], y[0], x[-1], y[-1]
    num = np.abs((y2 - y1) * x - (x2 - x1) * y + (x2 - x1) * y1 - (y2 - y1) * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    elbow_idx = int(np.argmax(num / (den + 1e-12)))
    return float(distances[elbow_idx])


def compute_silhouette(matrix, labels, max_sample=10_000):
    """
    Compute the silhouette score for a clustering result.
    Uses random sampling for large datasets to keep computation fast.
    Returns None if fewer than 2 unique labels are present.
    """
    from sklearn.metrics import silhouette_score
    if len(set(labels.tolist())) < 2:
        return None
    sample = min(max_sample, len(labels))
    return float(silhouette_score(matrix, labels, sample_size=sample, random_state=0))
