import numpy as np
from sklearn.decomposition import PCA


def run_pca(matrix, mode, value):
    """
    Apply PCA to the feature matrix.

    mode:
      'variance' — retain components that together explain `value` fraction of
                   total variance (e.g. value=0.90 keeps the 90% threshold)
      'fixed'    — retain exactly `value` components
      'none'     — skip PCA, return matrix unchanged

    Returns (transformed_matrix, pca_object_or_None, explained_variance_ratio_or_None)
    """
    if mode == "none":
        return matrix, None, None

    if mode == "variance":
        pca = PCA(n_components=float(value))
    else:
        n = min(int(value), matrix.shape[1], matrix.shape[0] - 1)
        n = max(1, n)
        pca = PCA(n_components=n)

    transformed = pca.fit_transform(matrix)
    return transformed, pca, pca.explained_variance_ratio_


def format_pca_text(pca, explained_variance_ratio):
    n = len(explained_variance_ratio)
    total = float(np.sum(explained_variance_ratio)) * 100
    lines = [f"Retained {n} component(s), explaining {total:.1f}% of total variance."]
    for i, ev in enumerate(explained_variance_ratio):
        lines.append(f"  PC{i + 1}: {ev * 100:.1f}%")
    return "\n".join(lines)
