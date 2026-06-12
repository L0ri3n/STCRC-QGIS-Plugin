import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = None


def write_classified_raster(labels, valid_mask, ref_profile, out_path):
    """
    Write a classified label raster preserving the reference georeferencing.

    Cluster labels occupy valid pixels; nodata (-9999) fills masked pixels.
    Labels are written as int16 so DBSCAN noise (-1) and IF labels (-1/1)
    are all represented exactly.
    """
    if rasterio is None:
        raise ImportError("rasterio is required for writing rasters.")

    profile = ref_profile.copy()
    profile.update({
        "dtype": "int16",
        "count": 1,
        "nodata": -9999,
        "compress": "lzw",
    })

    out = np.full(valid_mask.shape, -9999, dtype=np.int16)
    out[valid_mask] = labels.astype(np.int16)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)


def write_score_raster(scores, valid_mask, ref_profile, out_path):
    """
    Write a continuous anomaly score raster (Isolation Forest output).
    More negative score = more anomalous.
    """
    if rasterio is None:
        raise ImportError("rasterio is required for writing rasters.")

    profile = ref_profile.copy()
    profile.update({
        "dtype": "float32",
        "count": 1,
        "nodata": float("nan"),
        "compress": "lzw",
    })

    out = np.full(valid_mask.shape, np.nan, dtype=np.float32)
    out[valid_mask] = scores.astype(np.float32)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)


def plot_cluster_profiles(raster_tuples, labels, valid_mask, date_labels=None, out_path=None,
                          max_clusters=20, norm_method=None):
    """
    Plot mean temporal profile for each cluster label.

    date_labels: optional list of strings (one per raster) used as x-tick labels.
                 Falls back to "T1, T2, ..." when not provided or partially empty.
    max_clusters: if there are more non-noise cluster labels than this, only the
                  largest max_clusters clusters are plotted (noise is always included).
    If out_path is given, saves the figure there; otherwise shows it.
    Returns the figure object (caller is responsible for closing it).
    """
    import matplotlib.pyplot as plt

    arrays = [arr.data for arr, _ in raster_tuples]
    n_bands = len(arrays)
    all_labels = sorted(set(labels.tolist()))

    # Separate noise from real clusters; limit real clusters to the largest N by pixel count
    noise_labels = [l for l in all_labels if l == -1]
    cluster_labels = [l for l in all_labels if l != -1]
    if len(cluster_labels) > max_clusters:
        cluster_labels = sorted(cluster_labels,
                                key=lambda l: int(np.sum(labels == l)),
                                reverse=True)[:max_clusters]
    truncated = len(cluster_labels) < len([l for l in all_labels if l != -1])
    unique_labels = noise_labels + sorted(cluster_labels)

    # Build x-tick labels — use user dates where provided, index fallback otherwise
    if date_labels and any(d.strip() for d in date_labels):
        x_labels = [d.strip() if d.strip() else f"T{i+1}"
                    for i, d in enumerate(date_labels)]
    else:
        x_labels = [f"T{i+1}" for i in range(n_bands)]

    x_pos = range(n_bands)

    fig, ax = plt.subplots(figsize=(7, 3.5), tight_layout=True)

    for lbl in unique_labels:
        cluster_pixels = labels == lbl
        means = [arr[valid_mask][cluster_pixels].mean() for arr in arrays]
        name = "Noise† (DBSCAN)" if lbl == -1 else f"Cluster {lbl}"
        style = {"linestyle": "--", "alpha": 0.6} if lbl == -1 else {}
        ax.plot(x_pos, means, marker="o", label=name, **style)

    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(x_labels, rotation=30 if any(len(l) > 4 for l in x_labels) else 0,
                       ha="right", fontsize=8)
    ax.set_xlabel("Time step")
    if norm_method is None or norm_method == "None":
        y_label = "Mean value (unnormalised)" if norm_method == "None" else "Mean value (normalised)"
    else:
        y_label = f"Mean value (normalised — {norm_method})"
    ax.set_ylabel(y_label)
    title = "Cluster temporal profiles"
    if truncated:
        title += f"  (top {max_clusters} clusters by size shown)"
    if -1 in unique_labels:
        title += "\n† Noise = mean of all DBSCAN outlier pixels (not a coherent group)"
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None

    return fig


def compute_medoids(labels, matrix, valid_mask, ref_profile):
    """
    Find the spatial medoid of each cluster — the pixel whose feature vector is
    closest to the cluster centroid in the (PCA-transformed) feature space.

    Noise pixels (label -1) are excluded.
    Returns a list of dicts: {label, x, y, n_pixels} in the raster CRS.
    """
    import rasterio.transform as rtransform

    transform = ref_profile["transform"]
    rows, cols = np.where(valid_mask)
    cluster_labels = sorted(l for l in set(labels.tolist()) if l != -1)

    result = []
    for lbl in cluster_labels:
        mask = labels == lbl
        cluster_matrix = matrix[mask]
        centroid = cluster_matrix.mean(axis=0)
        dists = np.linalg.norm(cluster_matrix - centroid, axis=1)
        local_idx = int(np.argmin(dists))
        global_idx = int(np.where(mask)[0][local_idx])
        r, c = int(rows[global_idx]), int(cols[global_idx])
        x, y = rtransform.xy(transform, r, c)
        result.append({
            "label":    lbl,
            "x":        float(x),
            "y":        float(y),
            "n_pixels": int(np.sum(mask)),
        })

    return result
