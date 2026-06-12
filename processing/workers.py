from PyQt5.QtCore import QObject, pyqtSignal

from .stage1_ingestion import (
    load_rasters, compute_diagnostics, normalize, build_feature_matrix,
)
from .stage3_classification import (
    run_kmeans, run_dbscan, run_isolation_forest, compute_silhouette,
)


class Stage1Worker(QObject):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, paths, date_labels, nodata_override, align, norm_method, feat_type):
        super().__init__()
        self._paths           = paths
        self._date_labels     = date_labels
        self._nodata_override = nodata_override
        self._align           = align
        self._norm_method     = norm_method
        self._feat_type       = feat_type
        self._cancelled       = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self.progress.emit("Loading and validating rasters…")
            raster_tuples = load_rasters(
                self._paths,
                nodata_override=self._nodata_override,
                align=self._align,
            )
            if self._cancelled:
                return

            diagnostics = compute_diagnostics(raster_tuples)

            self.progress.emit("Normalizing…")
            normalized = normalize(raster_tuples, self._norm_method)
            if self._cancelled:
                return

            self.progress.emit("Building feature matrix…")
            matrix, valid_mask, ref_profile = build_feature_matrix(normalized, self._feat_type)

            self.finished.emit({
                "raster_tuples": normalized,
                "matrix":        matrix,
                "valid_mask":    valid_mask,
                "ref_profile":   ref_profile,
                "diagnostics":   diagnostics,
                "date_labels":   self._date_labels,
                "norm_method":   self._norm_method,
            })
        except Exception as e:
            self.error.emit(str(e))


class Stage3Worker(QObject):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, matrix, method, k, eps, min_samples, contamination):
        super().__init__()
        self._matrix        = matrix
        self._method        = method
        self._k             = k
        self._eps           = eps
        self._min_samples   = min_samples
        self._contamination = contamination
        self._cancelled     = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self.progress.emit(f"Running {self._method}…")
            if self._method == "K-Means":
                labels = run_kmeans(self._matrix, self._k)
                scores = None
                self.progress.emit("Computing silhouette score…")
                try:
                    sil = compute_silhouette(self._matrix, labels)
                except Exception:
                    sil = None
            elif self._method == "DBSCAN":
                labels = run_dbscan(self._matrix, self._eps, self._min_samples)
                scores = None
                sil    = None
            else:
                scores, labels = run_isolation_forest(self._matrix, self._contamination)
                sil = None

            self.finished.emit({"labels": labels, "scores": scores, "silhouette": sil})
        except Exception as e:
            self.error.emit(str(e))

