import os
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QTextEdit, QMessageBox, QCheckBox,
    QDoubleSpinBox, QSpinBox, QStackedWidget, QWidget,
    QScrollArea, QFrame, QApplication, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QProgressBar,
)
from PyQt5.QtCore import Qt, QThread

from .processing.stage1_ingestion import (
    load_rasters, compute_diagnostics, format_diagnostic_text,
    normalize, build_feature_matrix,
)
from .processing.stage2_pca import run_pca, format_pca_text
from .processing.stage3_classification import (
    run_kmeans, run_dbscan, run_isolation_forest,
    compute_knn_distances, detect_elbow, compute_silhouette,
)
from .processing.stage4_output import (
    write_classified_raster, write_score_raster, plot_cluster_profiles,
    compute_medoids,
)
from .processing.workers import Stage1Worker, Stage3Worker


# ---------------------------------------------------------------------------
# Reusable collapsible section widget

class CollapsibleSection(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self._open = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 0, 2)
        outer.setSpacing(0)

        self._header = QWidget()
        self._header.setStyleSheet(
            "QWidget { background-color: #d6dde6; border-radius: 3px; }"
            "QWidget:hover { background-color: #c2cdd9; }"
        )
        self._header.setCursor(Qt.PointingHandCursor)
        h_layout = QHBoxLayout(self._header)
        h_layout.setContentsMargins(6, 5, 6, 5)

        self._arrow = QLabel("▶")
        self._arrow.setFixedWidth(14)
        h_layout.addWidget(self._arrow)

        h_layout.addWidget(QLabel(f"<b>{title}</b>"))
        h_layout.addStretch()

        self._done = QLabel("")
        self._done.setStyleSheet("color: #2e7d32; font-size: 13px; font-weight: bold;")
        self._done.setFixedWidth(18)
        h_layout.addWidget(self._done)

        outer.addWidget(self._header)

        self._content = QWidget()
        c_outer = QVBoxLayout(self._content)
        c_outer.setContentsMargins(8, 6, 8, 8)
        self._inner = QVBoxLayout()
        self._inner.setSpacing(4)
        c_outer.addLayout(self._inner)
        self._content.setVisible(False)
        outer.addWidget(self._content)

        self._header.mousePressEvent = lambda _e: self.toggle()

    @property
    def content_layout(self):
        return self._inner

    def toggle(self):
        self.set_open(not self._open)

    def set_open(self, value):
        self._open = value
        self._content.setVisible(value)
        self._arrow.setText("▼" if value else "▶")

    def set_complete(self, value):
        self._done.setText("✓" if value else "")


# ---------------------------------------------------------------------------

class STCRCDialog(QDialog):
    def __init__(self, iface, parent=None):
        super().__init__(parent or iface.mainWindow())
        self.iface = iface
        self._raster_paths = []
        self._stage1_thread = None
        self._stage1_worker = None
        self._stage3_thread = None
        self._stage3_worker = None

        self.setWindowTitle("STCRC — Spatio-Temporal Change Regime Classification")
        self.setMinimumWidth(660)
        screen_h = QApplication.primaryScreen().availableGeometry().height()
        self.resize(680, min(700, screen_h - 80))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setSpacing(4)

        self._s1 = CollapsibleSection("Stage 1: Data Ingestion and Preparation")
        self._s2 = CollapsibleSection("Stage 2: PCA (Dimensionality Reduction)")
        self._s3 = CollapsibleSection("Stage 3: Classification")
        self._s4 = CollapsibleSection("Stage 4: Output")

        self._s1.set_open(True)

        self._populate_stage1(self._s1.content_layout)
        self._populate_stage2(self._s2.content_layout)
        self._populate_stage3(self._s3.content_layout)
        self._populate_stage4(self._s4.content_layout)

        for s in (self._s1, self._s2, self._s3, self._s4):
            cl.addWidget(s)
        cl.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)

        # Progress indicator — hidden when idle
        progress_row = QHBoxLayout()
        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("color: #2b6cb0; font-style: italic;")
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)   # indeterminate pulse
        self._progress_bar.setFixedHeight(14)
        self._progress_label.setVisible(False)
        self._progress_bar.setVisible(False)
        progress_row.addWidget(self._progress_label)
        progress_row.addWidget(self._progress_bar)
        outer.addLayout(progress_row)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        outer.addWidget(close_btn)

    # ------------------------------------------------------------------ Stage 1

    def _populate_stage1(self, layout):
        layout.addWidget(QLabel("Input rasters (ordered by time). Fill in the date/label column for readable profile plots:"))

        self._raster_table = QTableWidget(0, 2)
        self._raster_table.setHorizontalHeaderLabels(["Raster file", "Date / label"])
        self._raster_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._raster_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._raster_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._raster_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._raster_table.verticalHeader().setVisible(False)
        self._raster_table.setFixedHeight(120)
        layout.addWidget(self._raster_table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add raster(s)…")
        add_btn.clicked.connect(self._add_rasters)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_raster)
        up_btn = QPushButton("▲")
        up_btn.setFixedWidth(30)
        up_btn.clicked.connect(self._move_up)
        dn_btn = QPushButton("▼")
        dn_btn.setFixedWidth(30)
        dn_btn.clicked.connect(self._move_down)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch()
        btn_row.addWidget(up_btn)
        btn_row.addWidget(dn_btn)
        layout.addLayout(btn_row)

        self._align_check = QCheckBox("Auto-align to intersection (bilinear resampling)")
        self._align_check.setToolTip(
            "Clips all rasters to their common spatial overlap and resamples to a shared grid.\n"
            "Use when rasters have a 1-pixel size mismatch from clipping."
        )
        layout.addWidget(self._align_check)

        feat_row = QHBoxLayout()
        feat_row.addWidget(QLabel("Features:"))
        self._feat_combo = QComboBox()
        self._feat_combo.addItems(["Raw values", "Consecutive differences"])
        self._feat_combo.setToolTip(
            "Raw values: each pixel described by its value at every time step. Captures the full\n"
            "trajectory shape. Recommended when the absolute level of the phenomenon matters.\n\n"
            "Consecutive differences: each pixel described by its change between steps (N→N-1\n"
            "features). Captures rate of change rather than absolute level. Use when the\n"
            "magnitude of change matters more than the starting level."
        )
        feat_row.addWidget(self._feat_combo)
        feat_row.addStretch()
        layout.addLayout(feat_row)

        norm_row = QHBoxLayout()
        norm_row.addWidget(QLabel("Normalization:"))
        self._norm_combo = QComboBox()
        self._norm_combo.addItems(["Standard", "Robust", "Quantile", "MinMax", "None"])
        self._norm_combo.setToolTip(
            "Standard: zero mean, unit variance. Default for PSI and most datasets.\n"
            "Robust: uses median and IQR instead of mean/std. Use when outlier pixels\n"
            "  (sinkholes, edge artefacts) are present — check the diagnostic below.\n"
            "Quantile: maps any distribution to Gaussian output. Use for heavily skewed data.\n"
            "MinMax: scales to [0, 1]. Use only when all rasters share a meaningful common scale.\n"
            "None: skip normalization. Only safe when all rasters are already comparable."
        )
        norm_row.addWidget(self._norm_combo)
        norm_row.addStretch()
        layout.addLayout(norm_row)

        nodata_row = QHBoxLayout()
        self._nodata_override_check = QCheckBox("Override NoData value:")
        self._nodata_override_check.setToolTip(
            "Use when rasters have NoData pixels but the sentinel value is not stored in\n"
            "the file metadata. Common values: -9999, -32768, 0.0.\n"
            "Leave unchecked to rely on the raster's embedded NoData setting."
        )
        self._nodata_override_spin = QDoubleSpinBox()
        self._nodata_override_spin.setRange(-1e9, 1e9)
        self._nodata_override_spin.setDecimals(4)
        self._nodata_override_spin.setValue(-9999.0)
        self._nodata_override_spin.setEnabled(False)
        self._nodata_override_check.toggled.connect(self._nodata_override_spin.setEnabled)
        nodata_row.addWidget(self._nodata_override_check)
        nodata_row.addWidget(self._nodata_override_spin)
        nodata_row.addStretch()
        layout.addLayout(nodata_row)

        layout.addWidget(QLabel("Distribution diagnostic (skewness per band):"))
        self._diag_text = QTextEdit()
        self._diag_text.setReadOnly(True)
        self._diag_text.setFixedHeight(80)
        self._diag_text.setPlaceholderText("Load rasters to see per-band statistics and scaler suggestions.")
        layout.addWidget(self._diag_text)

        self._load_btn = QPushButton("Load & Diagnose")
        self._load_btn.clicked.connect(self._run_stage1)
        layout.addWidget(self._load_btn)

        self._stage1_status = QLabel("")
        self._stage1_status.setWordWrap(True)
        layout.addWidget(self._stage1_status)

    # ------------------------------------------------------------------ Stage 2

    def _populate_stage2(self, layout):
        self._pca_skip_check = QCheckBox("Skip PCA — pass features directly to classification")
        self._pca_skip_check.toggled.connect(self._on_pca_skip_toggled)
        layout.addWidget(self._pca_skip_check)

        self._pca_controls = QWidget()
        pca_cl = QVBoxLayout(self._pca_controls)
        pca_cl.setContentsMargins(0, 0, 0, 0)
        pca_cl.setSpacing(4)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._pca_mode_combo = QComboBox()
        self._pca_mode_combo.addItems(["Variance threshold (auto)", "Fixed n_components"])
        self._pca_mode_combo.currentIndexChanged.connect(self._on_pca_mode_changed)
        mode_row.addWidget(self._pca_mode_combo)
        mode_row.addStretch()
        pca_cl.addLayout(mode_row)

        self._pca_param_stack = QStackedWidget()

        var_page = QWidget()
        vl = QHBoxLayout(var_page)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.addWidget(QLabel("Variance to retain:"))
        self._pca_variance_spin = QDoubleSpinBox()
        self._pca_variance_spin.setRange(0.50, 0.99)
        self._pca_variance_spin.setSingleStep(0.05)
        self._pca_variance_spin.setValue(0.90)
        self._pca_variance_spin.setDecimals(2)
        vl.addWidget(self._pca_variance_spin)
        vl.addStretch()
        self._pca_param_stack.addWidget(var_page)

        fix_page = QWidget()
        fl = QHBoxLayout(fix_page)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.addWidget(QLabel("Number of components:"))
        self._pca_n_spin = QSpinBox()
        self._pca_n_spin.setRange(1, 99)
        self._pca_n_spin.setValue(2)
        fl.addWidget(self._pca_n_spin)
        fl.addStretch()
        self._pca_param_stack.addWidget(fix_page)

        pca_cl.addWidget(self._pca_param_stack)

        self._pca_result_text = QTextEdit()
        self._pca_result_text.setReadOnly(True)
        self._pca_result_text.setFixedHeight(65)
        self._pca_result_text.setPlaceholderText("Run PCA to see explained variance per component.")
        pca_cl.addWidget(self._pca_result_text)

        run_btn = QPushButton("Run PCA")
        run_btn.clicked.connect(self._run_stage2)
        pca_cl.addWidget(run_btn)

        layout.addWidget(self._pca_controls)

        self._stage2_status = QLabel("")
        self._stage2_status.setWordWrap(True)
        layout.addWidget(self._stage2_status)

    # ------------------------------------------------------------------ Stage 3

    _METHOD_HINTS = {
        "K-Means": (
            "Best for identifying dominant spatial regimes. Groups pixels whose temporal "
            "trajectories are most similar across all time steps. Use this when the question "
            "is 'what are the main types of behaviour in the scene?' — stable zones, "
            "linearly changing areas, accelerating areas, etc."
        ),
        "DBSCAN": (
            "Density-based clustering — finds clusters of any shape without a predefined k. "
            "Pixels in low-density regions are flagged as noise (label −1), which naturally "
            "highlights spatially isolated anomalies. Requires careful eps selection via the "
            "k-NN plot. Works best with larger time stacks (5+ rasters) where the higher-"
            "dimensional feature space develops the density structure DBSCAN needs. With only "
            "2–3 rasters the smooth feature space often produces many micro-clusters."
        ),
        "Isolation Forest": (
            "Anomaly detection, not clustering. Scores each pixel by how unusual its temporal "
            "profile is relative to the rest of the scene. Best for 'where is something weird "
            "happening?' — complex fluctuators, sudden events, early sinkhole precursors. "
            "Complements K-Means rather than replacing it: run K-Means first to map the "
            "dominant regimes, then Isolation Forest to catch what doesn't fit any of them."
        ),
    }

    def _populate_stage3(self, layout):
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self._cls_method_combo = QComboBox()
        self._cls_method_combo.addItems(["K-Means", "DBSCAN", "Isolation Forest"])
        self._cls_method_combo.currentIndexChanged.connect(self._on_cls_method_changed)
        method_row.addWidget(self._cls_method_combo)
        method_row.addStretch()
        layout.addLayout(method_row)

        self._method_hint_label = QLabel(self._METHOD_HINTS["K-Means"])
        self._method_hint_label.setWordWrap(True)
        self._method_hint_label.setStyleSheet("color: #4a5568; font-style: italic; font-size: 11px;")
        layout.addWidget(self._method_hint_label)

        self._cls_stack = QStackedWidget()

        km_page = QWidget()
        kl = QHBoxLayout(km_page)
        kl.setContentsMargins(0, 0, 0, 0)
        kl.addWidget(QLabel("Number of clusters (k):"))
        self._km_k_spin = QSpinBox()
        self._km_k_spin.setRange(2, 50)
        self._km_k_spin.setValue(3)
        kl.addWidget(self._km_k_spin)
        kl.addStretch()
        self._cls_stack.addWidget(km_page)

        db_page = QWidget()
        dl = QVBoxLayout(db_page)
        dl.setContentsMargins(0, 0, 0, 0)
        db_params = QHBoxLayout()
        db_params.addWidget(QLabel("min_samples:"))
        self._db_min_samples_spin = QSpinBox()
        self._db_min_samples_spin.setRange(2, 500)
        self._db_min_samples_spin.setValue(5)
        self._db_min_samples_spin.setToolTip(
            "Rule of thumb: n_PCA_components + 1, minimum 5.\n"
            "Recompute the k-NN plot if you change this."
        )
        db_params.addWidget(self._db_min_samples_spin)
        db_params.addSpacing(16)
        db_params.addWidget(QLabel("eps:"))
        self._db_eps_spin = QDoubleSpinBox()
        self._db_eps_spin.setRange(0.001, 9999.0)
        self._db_eps_spin.setDecimals(4)
        self._db_eps_spin.setValue(0.5)
        self._db_eps_spin.setToolTip("Neighbourhood radius. Pre-filled from elbow detection.")
        db_params.addWidget(self._db_eps_spin)
        db_params.addStretch()
        dl.addLayout(db_params)

        knn_btn = QPushButton("Compute k-NN distance plot")
        knn_btn.clicked.connect(self._run_knn_plot)
        dl.addWidget(knn_btn)

        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            self._knn_fig = Figure(figsize=(4, 1.8), dpi=80, tight_layout=True)
            self._knn_canvas = FigureCanvas(self._knn_fig)
            self._knn_canvas.setFixedHeight(145)
            self._knn_ax = self._knn_fig.add_subplot(111)
            dl.addWidget(self._knn_canvas)
        except Exception:
            self._knn_canvas = None
            dl.addWidget(QLabel("(matplotlib canvas unavailable — set eps manually)"))

        self._cls_stack.addWidget(db_page)

        iso_page = QWidget()
        il = QHBoxLayout(iso_page)
        il.setContentsMargins(0, 0, 0, 0)
        il.addWidget(QLabel("Contamination (expected anomaly fraction):"))
        self._iso_contamination_spin = QDoubleSpinBox()
        self._iso_contamination_spin.setRange(0.001, 0.5)
        self._iso_contamination_spin.setSingleStep(0.01)
        self._iso_contamination_spin.setDecimals(3)
        self._iso_contamination_spin.setValue(0.05)
        il.addWidget(self._iso_contamination_spin)
        il.addStretch()
        self._cls_stack.addWidget(iso_page)

        layout.addWidget(self._cls_stack)

        self._run3_btn = QPushButton("Run Classification")
        self._run3_btn.clicked.connect(self._run_stage3)
        layout.addWidget(self._run3_btn)

        self._stage3_status = QLabel("")
        self._stage3_status.setWordWrap(True)
        layout.addWidget(self._stage3_status)

    # ------------------------------------------------------------------ Stage 4

    def _populate_stage4(self, layout):
        raster_row = QHBoxLayout()
        raster_row.addWidget(QLabel("Output raster:"))
        self._out_raster_edit = QLineEdit()
        self._out_raster_edit.setPlaceholderText("Path for classified raster (.tif)")
        raster_row.addWidget(self._out_raster_edit)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_output_raster)
        raster_row.addWidget(browse_btn)
        layout.addLayout(raster_row)

        self._autoload_check = QCheckBox("Add result to QGIS canvas after writing")
        self._autoload_check.setChecked(True)
        layout.addWidget(self._autoload_check)

        self._profiles_check = QCheckBox("Export cluster profile plot (.png alongside raster)")
        self._profiles_check.setChecked(True)
        layout.addWidget(self._profiles_check)

        self._score_check = QCheckBox("Export anomaly score raster (Isolation Forest only)")
        self._score_check.setVisible(False)
        layout.addWidget(self._score_check)

        self._medoids_check = QCheckBox("Export cluster medoids as point layer (.shp)")
        self._medoids_check.setToolTip(
            "Writes a point shapefile with one feature per cluster, placed at the pixel\n"
            "whose temporal profile is closest to the cluster centroid (spatial medoid).\n"
            "Attributes: cluster_id, n_pixels. Not available for Isolation Forest."
        )
        layout.addWidget(self._medoids_check)

        medoids_path_row = QHBoxLayout()
        self._medoids_path_edit = QLineEdit()
        self._medoids_path_edit.setPlaceholderText("Path for medoids layer (.shp)")
        self._medoids_path_edit.setVisible(False)
        medoids_path_row.addWidget(self._medoids_path_edit)
        self._browse_medoids_btn = QPushButton("Browse…")
        self._browse_medoids_btn.setVisible(False)
        self._browse_medoids_btn.clicked.connect(self._browse_output_medoids)
        medoids_path_row.addWidget(self._browse_medoids_btn)
        layout.addLayout(medoids_path_row)

        self._medoids_check.toggled.connect(self._medoids_path_edit.setVisible)
        self._medoids_check.toggled.connect(self._browse_medoids_btn.setVisible)

        write_btn = QPushButton("Write Output")
        write_btn.clicked.connect(self._run_stage4)
        layout.addWidget(write_btn)

        self._stage4_status = QLabel("")
        self._stage4_status.setWordWrap(True)
        layout.addWidget(self._stage4_status)

    # ------------------------------------------------------------------ Slot helpers

    def _add_rasters(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select raster files (ordered by time)", "",
            "Raster files (*.tif *.tiff *.img *.vrt *.nc *.asc);;All files (*)",
        )
        for p in paths:
            if p not in self._raster_paths:
                self._raster_paths.append(p)
                row = self._raster_table.rowCount()
                self._raster_table.insertRow(row)
                name_item = QTableWidgetItem(os.path.basename(p))
                name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
                self._raster_table.setItem(row, 0, name_item)
                self._raster_table.setItem(row, 1, QTableWidgetItem(""))

    def _remove_raster(self):
        row = self._raster_table.currentRow()
        if row >= 0:
            self._raster_table.removeRow(row)
            self._raster_paths.pop(row)

    def _swap_rows(self, a, b):
        for col in range(self._raster_table.columnCount()):
            item_a = self._raster_table.takeItem(a, col)
            item_b = self._raster_table.takeItem(b, col)
            self._raster_table.setItem(a, col, item_b)
            self._raster_table.setItem(b, col, item_a)
        self._raster_paths[a], self._raster_paths[b] = self._raster_paths[b], self._raster_paths[a]

    def _move_up(self):
        row = self._raster_table.currentRow()
        if row > 0:
            self._swap_rows(row, row - 1)
            self._raster_table.setCurrentCell(row - 1, 1)

    def _move_down(self):
        row = self._raster_table.currentRow()
        if 0 <= row < self._raster_table.rowCount() - 1:
            self._swap_rows(row, row + 1)
            self._raster_table.setCurrentCell(row + 1, 1)

    def _get_date_labels(self):
        labels = []
        for row in range(self._raster_table.rowCount()):
            item = self._raster_table.item(row, 1)
            labels.append(item.text().strip() if item else "")
        return labels

    def _on_pca_mode_changed(self, index):
        self._pca_param_stack.setCurrentIndex(index)

    def _on_pca_skip_toggled(self, checked):
        self._pca_controls.setVisible(not checked)
        if checked:
            if hasattr(self, "_stage1_result"):
                self._apply_pca_skip()
        else:
            if hasattr(self, "_stage2_result"):
                del self._stage2_result
            self._stage2_status.setText("")
            self._s2.set_complete(False)

    def _apply_pca_skip(self):
        r = self._stage1_result
        self._stage2_result = {
            "matrix": r["matrix"], "pca": None, "explained_variance_ratio": None,
            "valid_mask": r["valid_mask"], "ref_profile": r["ref_profile"],
            "raster_tuples": r["raster_tuples"], "date_labels": r["date_labels"],
            "norm_method": r["norm_method"],
        }
        n = r["matrix"].shape[1]
        self._stage2_status.setText(f"PCA skipped — {n} feature(s) passed to classification.")
        self._s2.set_complete(True)
        self._s3.set_open(True)

    def _on_cls_method_changed(self, index):
        self._cls_stack.setCurrentIndex(index)
        method = self._cls_method_combo.currentText()
        self._method_hint_label.setText(self._METHOD_HINTS.get(method, ""))
        is_iso = method == "Isolation Forest"
        self._profiles_check.setVisible(not is_iso)
        self._score_check.setVisible(is_iso)
        self._medoids_check.setVisible(not is_iso)
        if is_iso and self._medoids_check.isChecked():
            self._medoids_check.setChecked(False)

    def _browse_output_raster(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self._out_raster_edit.setText(os.path.join(folder, "STCRC_classified.tif"))

    def _browse_output_medoids(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder for medoids layer")
        if folder:
            self._medoids_path_edit.setText(os.path.join(folder, "STCRC_medoids.shp"))

    # ------------------------------------------------------------------ Progress helpers

    def _show_progress(self, msg=""):
        self._progress_label.setText(msg)
        self._progress_label.setVisible(True)
        self._progress_bar.setVisible(True)

    def _hide_progress(self):
        self._progress_label.setVisible(False)
        self._progress_bar.setVisible(False)
        self._progress_label.setText("")

    def _cleanup_threads(self):
        for attr_w, attr_t in [("_stage1_worker", "_stage1_thread"),
                                ("_stage3_worker", "_stage3_thread")]:
            worker = getattr(self, attr_w, None)
            thread = getattr(self, attr_t, None)
            if worker is not None:
                worker.cancel()
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait(3000)

    # ------------------------------------------------------------------ Date validation

    @staticmethod
    def _parse_date_label(text):
        text = text.strip()
        if not text:
            return None
        if text.isdigit() and len(text) == 4:
            return datetime(int(text), 1, 1)
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y",
                    "%Y-%m", "%b %Y", "%B %Y", "%d %b %Y", "%d %B %Y"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    # ------------------------------------------------------------------ Stage runners

    def _run_stage1(self):
        self._diag_text.clear()
        self._stage1_status.setText("")

        if len(self._raster_paths) < 2:
            QMessageBox.warning(self, "STCRC", "Please add at least 2 rasters.")
            return

        feat_type = self._feat_combo.currentText()
        if feat_type == "Consecutive differences" and len(self._raster_paths) < 3:
            QMessageBox.warning(self, "STCRC",
                "Consecutive differences require at least 3 rasters (produces N−1 features).")
            return

        date_labels = self._get_date_labels()
        parsed = [self._parse_date_label(l) for l in date_labels]
        parseable = [p for p in parsed if p is not None]
        if len(parseable) >= 2 and parseable != sorted(parseable):
            QMessageBox.warning(
                self, "STCRC — Date order",
                "Date labels appear to be out of chronological order.\n\n"
                "The profile plot x-axis and 'Consecutive differences' feature mode "
                "both assume rasters are ordered from earliest to latest. "
                "Reorder the rasters or correct the labels before proceeding."
            )

        nodata_override = (
            self._nodata_override_spin.value()
            if self._nodata_override_check.isChecked()
            else None
        )

        self._load_btn.setEnabled(False)
        self._show_progress("Loading rasters…")

        self._stage1_worker = Stage1Worker(
            list(self._raster_paths), date_labels, nodata_override,
            self._align_check.isChecked(), self._norm_combo.currentText(), feat_type,
        )
        self._stage1_thread = QThread(self)
        self._stage1_worker.moveToThread(self._stage1_thread)
        self._stage1_thread.started.connect(self._stage1_worker.run)
        self._stage1_worker.progress.connect(self._on_stage1_progress)
        self._stage1_worker.finished.connect(self._on_stage1_finished)
        self._stage1_worker.error.connect(self._on_stage1_error)
        self._stage1_worker.finished.connect(self._stage1_thread.quit)
        self._stage1_worker.error.connect(self._stage1_thread.quit)
        self._stage1_thread.finished.connect(self._stage1_worker.deleteLater)
        self._stage1_thread.start()

    def _on_stage1_progress(self, msg):
        self._progress_label.setText(msg)

    def _on_stage1_finished(self, result):
        self._hide_progress()
        self._load_btn.setEnabled(True)

        self._diag_text.setPlainText(format_diagnostic_text(result["diagnostics"]))

        self._stage1_result = {
            "raster_tuples": result["raster_tuples"],
            "matrix":        result["matrix"],
            "valid_mask":    result["valid_mask"],
            "ref_profile":   result["ref_profile"],
            "date_labels":   result["date_labels"],
            "norm_method":   result["norm_method"],
        }

        n_pixels, n_features = result["matrix"].shape
        self._stage1_status.setText(f"Ready: {n_pixels:,} valid pixels × {n_features} features.")
        self._s1.set_complete(True)
        self._s2.set_open(True)
        if self._pca_skip_check.isChecked():
            self._apply_pca_skip()

    def _on_stage1_error(self, msg):
        self._hide_progress()
        self._load_btn.setEnabled(True)
        QMessageBox.critical(self, "STCRC — Load error", msg)

    def _run_stage2(self):
        if self._pca_skip_check.isChecked():
            if not hasattr(self, "_stage1_result"):
                QMessageBox.warning(self, "STCRC", "Run Stage 1 first.")
                return
            self._apply_pca_skip()
            return

        self._pca_result_text.clear()
        self._stage2_status.setText("")

        if not hasattr(self, "_stage1_result"):
            QMessageBox.warning(self, "STCRC", "Run Stage 1 first.")
            return

        idx = self._pca_mode_combo.currentIndex()
        if idx == 0:
            mode, value = "variance", self._pca_variance_spin.value()
        else:
            mode, value = "fixed", self._pca_n_spin.value()

        try:
            transformed, pca_obj, evr = run_pca(self._stage1_result["matrix"], mode, value)
        except Exception as e:
            QMessageBox.critical(self, "STCRC — PCA error", str(e))
            return

        self._stage2_result = {
            "matrix": transformed, "pca": pca_obj, "explained_variance_ratio": evr,
            "valid_mask": self._stage1_result["valid_mask"],
            "ref_profile": self._stage1_result["ref_profile"],
            "raster_tuples": self._stage1_result["raster_tuples"],
            "date_labels": self._stage1_result["date_labels"],
            "norm_method": self._stage1_result["norm_method"],
        }

        self._pca_result_text.setPlainText(format_pca_text(pca_obj, evr))
        self._stage2_status.setText(f"Ready: {transformed.shape[1]} PC(s) passed to classification.")
        self._s2.set_complete(True)
        self._s3.set_open(True)

    def _run_knn_plot(self):
        if not hasattr(self, "_stage2_result"):
            QMessageBox.warning(self, "STCRC", "Run Stages 1 and 2 first.")
            return

        k = self._db_min_samples_spin.value()
        try:
            distances = compute_knn_distances(self._stage2_result["matrix"], k)
            suggested_eps = detect_elbow(distances)
        except Exception as e:
            QMessageBox.critical(self, "STCRC — k-NN error", str(e))
            return

        self._db_eps_spin.setValue(round(suggested_eps, 4))

        if self._knn_canvas is not None:
            self._knn_ax.clear()
            self._knn_ax.plot(distances, linewidth=1, color="#2b6cb0")
            self._knn_ax.axhline(y=suggested_eps, color="#e53e3e", linewidth=1,
                                  linestyle="--", label=f"eps ≈ {suggested_eps:.4f}")
            self._knn_ax.set_xlabel("Points (sorted)", fontsize=7)
            self._knn_ax.set_ylabel(f"{k}-NN distance", fontsize=7)
            self._knn_ax.tick_params(labelsize=6)
            self._knn_ax.legend(fontsize=6)
            self._knn_ax.yaxis.grid(True, alpha=0.4, linewidth=0.5)
            self._knn_ax.set_axisbelow(True)
            self._knn_canvas.draw()

        self._stage3_status.setText(
            f"Suggested eps = {suggested_eps:.4f}  (pre-filled above, edit if needed).")

    def _run_stage3(self):
        self._stage3_status.setText("")

        if not hasattr(self, "_stage2_result"):
            QMessageBox.warning(self, "STCRC", "Run Stages 1 and 2 first.")
            return

        method = self._cls_method_combo.currentText()

        self._run3_btn.setEnabled(False)
        self._show_progress(f"Running {method}…")

        self._stage3_worker = Stage3Worker(
            self._stage2_result["matrix"],
            method,
            k=self._km_k_spin.value(),
            eps=self._db_eps_spin.value(),
            min_samples=self._db_min_samples_spin.value(),
            contamination=self._iso_contamination_spin.value(),
        )
        self._stage3_thread = QThread(self)
        self._stage3_worker.moveToThread(self._stage3_thread)
        self._stage3_thread.started.connect(self._stage3_worker.run)
        self._stage3_worker.progress.connect(self._on_stage3_progress)
        self._stage3_worker.finished.connect(self._on_stage3_finished)
        self._stage3_worker.error.connect(self._on_stage3_error)
        self._stage3_worker.finished.connect(self._stage3_thread.quit)
        self._stage3_worker.error.connect(self._stage3_thread.quit)
        self._stage3_thread.finished.connect(self._stage3_worker.deleteLater)
        self._stage3_thread.start()

    def _on_stage3_progress(self, msg):
        self._progress_label.setText(msg)

    def _on_stage3_finished(self, result):
        self._hide_progress()
        self._run3_btn.setEnabled(True)

        labels = result["labels"]
        scores = result["scores"]
        sil    = result["silhouette"]
        method = self._cls_method_combo.currentText()

        if method == "K-Means":
            sil_str = f"  Silhouette: {sil:.3f}" if sil is not None else ""
            summary = f"K-Means done: {len(set(labels.tolist()))} clusters found.{sil_str}"
        elif method == "DBSCAN":
            n_noise = int(np.sum(labels == -1))
            summary = (f"DBSCAN done: {len(set(labels.tolist()) - {-1})} cluster(s), "
                       f"{n_noise:,} noise pixels.")
        else:
            summary = f"Isolation Forest done: {int(np.sum(labels == -1)):,} anomalous pixels."

        self._stage3_result = {
            "labels":        labels,
            "scores":        scores,
            "method":        method,
            "matrix":        self._stage2_result["matrix"],
            "valid_mask":    self._stage2_result["valid_mask"],
            "ref_profile":   self._stage2_result["ref_profile"],
            "raster_tuples": self._stage2_result["raster_tuples"],
            "date_labels":   self._stage2_result["date_labels"],
            "norm_method":   self._stage2_result.get("norm_method"),
        }

        self._stage3_status.setText(summary + " Proceed to Stage 4.")
        self._s3.set_complete(True)
        self._s4.set_open(True)

    def _on_stage3_error(self, msg):
        self._hide_progress()
        self._run3_btn.setEnabled(True)
        QMessageBox.critical(self, "STCRC — Classification error", msg)

    def _run_stage4(self):
        self._stage4_status.setText("")

        if not hasattr(self, "_stage3_result"):
            QMessageBox.warning(self, "STCRC", "Run Stages 1–3 first.")
            return

        out_path = self._out_raster_edit.text().strip()
        if not out_path:
            QMessageBox.warning(self, "STCRC", "Please set an output raster path.")
            return

        r = self._stage3_result
        messages = []

        try:
            write_classified_raster(r["labels"], r["valid_mask"], r["ref_profile"], out_path)
            messages.append("Raster written.")
        except Exception as e:
            QMessageBox.critical(self, "STCRC — Write error", str(e))
            return

        if self._autoload_check.isChecked():
            try:
                from qgis.core import QgsRasterLayer, QgsProject
                layer = QgsRasterLayer(out_path, "STCRC_" + r["method"].replace(" ", "_"))
                if layer.isValid():
                    QgsProject.instance().addMapLayer(layer)
                    messages.append("Layer added to canvas.")
                else:
                    messages.append("Written but could not auto-load into canvas.")
            except Exception as e:
                messages.append(f"Auto-load failed: {e}")

        if self._profiles_check.isChecked() and r["method"] != "Isolation Forest":
            profile_path = out_path.replace(".tif", "_profiles.png")
            try:
                plot_cluster_profiles(r["raster_tuples"], r["labels"], r["valid_mask"],
                                      date_labels=r.get("date_labels"),
                                      norm_method=r.get("norm_method"),
                                      out_path=profile_path)
                messages.append("Profile plot saved.")
            except Exception as e:
                messages.append(f"Profile plot failed: {e}")

        if self._score_check.isChecked() and r["method"] == "Isolation Forest":
            if r["scores"] is not None:
                score_path = out_path.replace(".tif", "_anomaly_score.tif")
                try:
                    write_score_raster(r["scores"], r["valid_mask"], r["ref_profile"], score_path)
                    messages.append("Score raster written.")
                except Exception as e:
                    messages.append(f"Score raster failed: {e}")

        if self._medoids_check.isChecked() and r["method"] != "Isolation Forest":
            medoids_path = self._medoids_path_edit.text().strip()
            if not medoids_path:
                messages.append("Medoids path not set — skipped.")
            else:
                try:
                    from qgis.core import (
                        QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY,
                        QgsProject, QgsField, QgsVectorFileWriter,
                        QgsCoordinateReferenceSystem,
                    )
                    from PyQt5.QtCore import QVariant

                    medoids = compute_medoids(r["labels"], r["matrix"],
                                             r["valid_mask"], r["ref_profile"])

                    src_crs = r["ref_profile"].get("crs")
                    crs = QgsCoordinateReferenceSystem()
                    if src_crs:
                        try:
                            epsg = src_crs.to_epsg()
                            if epsg:
                                crs = QgsCoordinateReferenceSystem(f"EPSG:{epsg}")
                            else:
                                crs = QgsCoordinateReferenceSystem(src_crs.to_wkt())
                        except Exception:
                            pass

                    crs_str = crs.authid() if crs.isValid() and crs.authid() else ""
                    mem_uri = f"Point?crs={crs_str}" if crs_str else "Point"
                    mem_layer = QgsVectorLayer(mem_uri, "STCRC_medoids", "memory")
                    dp = mem_layer.dataProvider()
                    dp.addAttributes([
                        QgsField("cluster_id", QVariant.Int),
                        QgsField("n_pixels",   QVariant.Int),
                    ])
                    mem_layer.updateFields()

                    features = []
                    for m in medoids:
                        feat = QgsFeature(mem_layer.fields())
                        feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(m["x"], m["y"])))
                        feat["cluster_id"] = m["label"]
                        feat["n_pixels"]   = m["n_pixels"]
                        features.append(feat)
                    dp.addFeatures(features)

                    error = QgsVectorFileWriter.writeAsVectorFormat(
                        mem_layer, medoids_path, "UTF-8", crs, "ESRI Shapefile"
                    )
                    if error[0] == QgsVectorFileWriter.NoError:
                        messages.append(f"Medoids written ({len(medoids)} points).")
                        if self._autoload_check.isChecked():
                            vlayer = QgsVectorLayer(medoids_path, "STCRC_medoids", "ogr")
                            if vlayer.isValid():
                                QgsProject.instance().addMapLayer(vlayer)
                    else:
                        messages.append(f"Medoids write failed: {error[1]}")
                except Exception as e:
                    messages.append(f"Medoids export failed: {e}")

        self._stage4_status.setText("  |  ".join(messages))
        self._s4.set_complete(True)

    # ------------------------------------------------------------------ Cleanup

    def closeEvent(self, event):
        self._cleanup_threads()
        super().closeEvent(event)

    def reject(self):
        self._cleanup_threads()
        super().reject()
