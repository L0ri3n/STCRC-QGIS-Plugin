def classFactory(iface):
    missing = []

    try:
        import rasterio
    except ImportError:
        missing.append("rasterio")

    try:
        import sklearn
        from packaging.version import Version
        if Version(sklearn.__version__) >= Version("1.6"):
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
        pkgs = "  " + "\n  ".join(missing)
        QMessageBox.critical(
            None,
            "STCRC — missing dependencies",
            "The following packages are required but not installed:\n\n"
            f"{pkgs}\n\n"
            "Install them by opening the OSGeo4W Shell and running:\n\n"
            "    pip install rasterio \"scikit-learn==1.5.2\"\n\n"
            "Then restart QGIS.",
        )
        return None

    from .plugin import STCRCPlugin
    return STCRCPlugin(iface)
