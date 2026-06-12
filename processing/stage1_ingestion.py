import numpy as np
import numpy.ma as ma
from scipy.stats import skew as scipy_skew

try:
    import rasterio
except ImportError:
    rasterio = None

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler


def _require_rasterio():
    if rasterio is None:
        raise ImportError(
            "rasterio is not installed. Open the OSGeo4W Shell and run: pip install rasterio"
        )


def load_rasters(paths, nodata_override=None, align=False):
    """
    Load N rasters and return list of (masked_array_2d, profile) tuples.

    When align=False (default): validates that all rasters share the same CRS,
    shape, and transform — raises ValueError with a descriptive message on mismatch.

    When align=True: reprojects all rasters to their spatial intersection using
    bilinear resampling. No raster is enlarged; all are cropped to the common
    overlap area. CRS mismatch is still a hard error — reproject before loading.
    """
    _require_rasterio()
    from rasterio.warp import reproject, Resampling
    import rasterio.transform as rtransform

    # First pass: collect metadata without reading pixel data
    meta_list = []
    for path in paths:
        with rasterio.open(path) as src:
            meta_list.append({
                "path": path,
                "crs": src.crs,
                "transform": src.transform,
                "bounds": src.bounds,
                "nodata": src.nodata,
                "res": src.res,
            })

    # CRS validation is always required
    ref_crs = meta_list[0]["crs"]
    ref_path = meta_list[0]["path"]
    for m in meta_list[1:]:
        if m["crs"] != ref_crs:
            raise ValueError(
                f"CRS mismatch: '{m['path']}' has CRS {m['crs']}, "
                f"but '{ref_path}' has CRS {ref_crs}. "
                "Reproject all rasters to a common CRS before loading."
            )

    # Compute intersection bounds and target grid when aligning
    if align:
        left   = max(m["bounds"].left   for m in meta_list)
        bottom = max(m["bounds"].bottom for m in meta_list)
        right  = min(m["bounds"].right  for m in meta_list)
        top    = min(m["bounds"].top    for m in meta_list)

        if left >= right or bottom >= top:
            raise ValueError(
                "Rasters have no spatial overlap. Check that all inputs cover a common area."
            )

        x_res, y_res = meta_list[0]["res"]
        dst_width  = max(1, int(round((right  - left)   / x_res)))
        dst_height = max(1, int(round((top    - bottom)  / y_res)))
        dst_transform = rtransform.from_bounds(left, bottom, right, top, dst_width, dst_height)
        dst_shape = (dst_height, dst_width)

    # Second pass: load and optionally align
    results = []
    ref_transform = meta_list[0]["transform"]
    ref_shape = None

    for i, (path, meta) in enumerate(zip(paths, meta_list)):
        with rasterio.open(path) as src:
            profile = src.profile.copy()
            data = src.read(1).astype(np.float64)
            nodata = nodata_override if nodata_override is not None else src.nodata

            if align:
                aligned = np.empty(dst_shape, dtype=np.float64)
                fill = nodata if nodata is not None else np.nan
                aligned[:] = fill
                reproject(
                    source=data,
                    destination=aligned,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=nodata,
                    dst_nodata=fill,
                )
                data = aligned
                profile.update({"height": dst_height, "width": dst_width, "transform": dst_transform})
            else:
                if i == 0:
                    ref_shape = data.shape
                else:
                    if data.shape != ref_shape:
                        raise ValueError(
                            f"Shape mismatch: '{path}' is {data.shape}, "
                            f"but '{ref_path}' is {ref_shape}. "
                            "Enable 'Auto-align to intersection' to handle this automatically."
                        )
                    if not np.allclose(
                        [src.transform.a, src.transform.b, src.transform.c,
                         src.transform.d, src.transform.e, src.transform.f],
                        [ref_transform.a, ref_transform.b, ref_transform.c,
                         ref_transform.d, ref_transform.e, ref_transform.f],
                        atol=1e-6,
                    ):
                        raise ValueError(
                            f"Transform mismatch: '{path}' has a different geotransform "
                            f"than '{ref_path}'. Enable 'Auto-align to intersection' or "
                            "ensure all rasters share the same pixel grid."
                        )

            nodata_val = nodata if nodata is not None else None
            if nodata_val is not None:
                pixel_mask = (data == nodata_val) | np.isnan(data)
            else:
                pixel_mask = np.isnan(data)

            results.append((ma.masked_array(data, mask=pixel_mask), profile))

    return results


def compute_diagnostics(raster_tuples):
    """
    Compute skewness per band and suggest a normalization method.

    Returns list of dicts: {band, skewness, outlier_pct, suggestion}.
    """
    diagnostics = []

    for i, (arr, _) in enumerate(raster_tuples):
        valid = arr.compressed()
        if valid.size == 0:
            diagnostics.append({
                "band": i + 1,
                "skewness": float("nan"),
                "outlier_pct": float("nan"),
                "suggestion": "No valid pixels",
            })
            continue

        sk = float(scipy_skew(valid))
        mean = valid.mean()
        std = valid.std()
        outlier_pct = float(np.sum(np.abs(valid - mean) > 3 * std) / len(valid) * 100)

        if abs(sk) > 1.5:
            suggestion = "Quantile"
        elif outlier_pct > 1.0:
            suggestion = "Robust"
        else:
            suggestion = "Standard"

        diagnostics.append({
            "band": i + 1,
            "skewness": sk,
            "outlier_pct": outlier_pct,
            "suggestion": suggestion,
        })

    return diagnostics


def format_diagnostic_text(diagnostics):
    """Format diagnostics list into a human-readable string for the UI."""
    lines = []
    for d in diagnostics:
        if d["suggestion"] == "No valid pixels":
            lines.append(f"Band {d['band']}: no valid pixels — check NoData settings.")
            continue
        lines.append(
            f"Band {d['band']}:  skewness = {d['skewness']:+.2f},  "
            f"outliers >3σ = {d['outlier_pct']:.1f}%  →  suggested scaler: {d['suggestion']}"
        )
    return "\n".join(lines)


def normalize(raster_tuples, method):
    """
    Normalize each band in place and return list of (normalized_masked_array, profile).

    method: 'Standard' | 'Robust' | 'Quantile' | 'MinMax' | 'None'
    """
    if method == "None":
        return raster_tuples

    scaler_map = {
        "Standard": StandardScaler,
        "Robust": RobustScaler,
        "MinMax": MinMaxScaler,
    }

    results = []
    for arr, profile in raster_tuples:
        valid_flat = arr.compressed().reshape(-1, 1)

        if method == "Quantile":
            scaler = QuantileTransformer(output_distribution="normal", random_state=0)
        else:
            scaler = scaler_map[method]()

        scaler.fit(valid_flat)

        out = arr.copy()
        out.data[~arr.mask] = scaler.transform(valid_flat).ravel()
        results.append((out, profile))

    return results


def build_feature_matrix(raster_tuples, feature_type):
    """
    Build a 2D feature matrix from a list of (masked_array, profile) tuples.

    feature_type: 'Raw values' | 'Consecutive differences'

    Returns (matrix, valid_mask, ref_profile) where:
      matrix.shape == (n_valid_pixels, n_features)
      valid_mask.shape == raster shape (2D bool, True = pixel used)
      ref_profile = profile from the first raster (for output georef)
    """
    arrays = [arr for arr, _ in raster_tuples]
    ref_profile = raster_tuples[0][1]

    combined_mask = np.zeros(arrays[0].shape, dtype=bool)
    for arr in arrays:
        combined_mask |= arr.mask

    valid_mask = ~combined_mask

    if feature_type == "Raw values":
        cols = [arr.data[valid_mask] for arr in arrays]
    else:
        diffs = [arrays[t + 1].data - arrays[t].data for t in range(len(arrays) - 1)]
        cols = [d[valid_mask] for d in diffs]

    matrix = np.column_stack(cols)
    return matrix, valid_mask, ref_profile
