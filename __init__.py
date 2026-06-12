import sys

# QGIS embeds Python with sys.stderr = None in some contexts.
# numpy and other C extensions try to write warnings there, which crashes.
if sys.stderr is None:
    import io
    sys.stderr = io.StringIO()


def classFactory(iface):
    missing = []

    try:
        import rasterio  # noqa: F401
    except ImportError:
        missing.append("rasterio")

    try:
        import sklearn
        # Avoid importing 'packaging' — not guaranteed in OSGeo4W.
        # Parse major.minor directly from the version string.
        parts = sklearn.__version__.split(".")
        major, minor = int(parts[0]), int(parts[1])
        if (major, minor) >= (1, 6):
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(
                None,
                "STCRC — dependency version warning",
                "scikit-learn 1.6+ is installed but may conflict with QGIS's bundled "
                "pyarrow/numpy. If you encounter errors, reinstall the pinned version:\n\n"
                "    pip install \"scikit-learn==1.5.2\"\n\n"
                "(run this in the OSGeo4W Shell)",
            )
    except ImportError:
        missing.append("scikit-learn==1.5.2")

    if missing:
        from PyQt5.QtWidgets import QMessageBox
        pkgs = "\n  ".join(missing)
        QMessageBox.critical(
            None,
            "STCRC — missing dependencies",
            "The following packages are required but not installed:\n\n"
            f"  {pkgs}\n\n"
            "Install them by opening the OSGeo4W Shell and running:\n\n"
            "    pip install rasterio \"scikit-learn==1.5.2\"\n\n"
            "Then restart QGIS.",
        )
        return None

    from .plugin import STCRCPlugin
    return STCRCPlugin(iface)
