
import numpy as np
import xarray as xr
from cmethods import adjust
import re

def get_forecast_columns(df):
    """GCM forecast columns eligible for bias correction (e.g. 'ecmwf_0_pr')."""
    return [c for c in df.columns if re.match(r"^(ecmwf|access|ukmo)_\d+_pr$", c)]


def bias_correct_with_cmethods(X_train, X_val, y_train, method="quantile_mapping",
                                kind="+", n_quantiles=80,
                                exclude_percentile=95.0, extrapolate_cutoff = 10.0, extrapolate_extremes=False):
    """
    Bias-correct each GCM forecast column in both X_train and X_val against
    y_train (observations), using X_train's matching column as the historical
    simulation (simh) in both cases.

    The quantile mapping is FIT only on non-extreme obs/simh pairs: a training
    row is excluded from fitting if either its obs value or its simh value
    exceeds its own `exclude_percentile`-th percentile. This stops a small
    number of extreme observed/forecast values from distorting the quantile
    mapping applied across the whole (mostly non-extreme) distribution.

    What happens to extreme values when APPLYING the fitted mapping is
    controlled by `extrapolate_extremes`:
      - False (default): values above the simh extreme threshold are left
        uncorrected (identity) rather than extrapolated through a mapping
        that was never fit on extreme data.
      - True: extreme values are corrected via LINEAR EXTRAPOLATION of the
        correction function's trend at the edge of the fitted (non-extreme)
        simh range, rather than being passed into adjust() directly. Two
        points are formed by evaluating the fitted mapping at slope 1
        ; a line is extended linearly beyond simh_threshold.

    Correcting X_train too (against itself) keeps the distribution XGBoost
    trains on consistent with what it sees at inference time on X_val.
    Builds xarray.DataArrays directly from the pandas data, since
    python-cmethods requires xarray.DataArray inputs rather than raw pandas
    objects.
    """
    X_train_corrected = X_train.copy()
    X_val_corrected = X_val.copy()

    # Minimum number of (obs, simh) pairs required before we trust a fitted
    # quantile mapping. Below this, a column (e.g. one with heavy NaNs,
    # like ukmo at short lead times) is left entirely uncorrected instead
    # of erroring out or fitting on a handful of noisy points.
    min_fit_points = max(10, n_quantiles)

    # Threshold defined on obs (y_train). Use nanpercentile defensively,
    # though y_train (post dropna on "pr") should not contain NaNs.
    obs_threshold = np.nanpercentile(y_train.values, exclude_percentile)

    def to_1d_array(corrected, var_name):
        """adjust() may return a DataArray or a Dataset depending on the
        python-cmethods version installed; normalize either to a flat
        numpy array."""
        if isinstance(corrected, xr.Dataset):
            if var_name in corrected.data_vars:
                corrected = corrected[var_name]
            else:
                corrected = corrected[list(corrected.data_vars)[0]]
        return np.asarray(corrected).ravel()

    for col in get_forecast_columns(X_train):

        simh_full = X_train[col].values

        # GCM columns (e.g. ukmo at short lead times) can be heavily
        # NaN. np.percentile propagates NaN -> NaN threshold -> every
        # comparison silently evaluates False -> empty fit/apply arrays
        # -> adjust() crashes on nanmax of a zero-size array. Use
        # nanpercentile and require enough *valid, non-extreme* pairs
        # before attempting to fit at all.
        valid_mask = ~np.isnan(y_train.values) & ~np.isnan(simh_full)
        n_valid = int(valid_mask.sum())

        if n_valid < min_fit_points:
            print(f"[bias_correct] '{col}': only {n_valid} valid (non-NaN) "
                  f"obs/simh pairs (< {min_fit_points}); skipping correction, "
                  "leaving column unmodified.")
            continue

        simh_threshold = np.nanpercentile(simh_full[valid_mask], exclude_percentile)

        # Exclude a training pair from the FIT if either side is extreme
        # or missing.
        non_extreme_mask = valid_mask & (y_train.values <= obs_threshold) & (simh_full <= simh_threshold)

        obs_fit = y_train.values[non_extreme_mask]
        simh_fit = simh_full[non_extreme_mask]

        if len(obs_fit) < min_fit_points:
            print(f"[bias_correct] '{col}': only {len(obs_fit)} non-extreme "
                  f"pairs available for fitting (< {min_fit_points}); "
                  "skipping correction, leaving column unmodified.")
            continue

        obs_da = xr.DataArray(
            obs_fit, dims="time",
            coords={"time": np.arange(len(obs_fit))}, name="obs_pr",
        )
        simh_fit_da = xr.DataArray(
            simh_fit, dims="time",
            coords={"time": np.arange(len(simh_fit))}, name=col,
        )

        def apply_correction(simp_values):
            simp_values = simp_values.copy()

            # NaNs in simp can't be passed through adjust(); carve them
            # out up front and restore them (as NaN) afterwards.
            simp_nan_mask = np.isnan(simp_values)
            simp_corrected = simp_values.copy()

            simp_extreme_mask = simp_nan_mask | (simp_values > simh_threshold)
            non_extreme_target_mask = ~simp_extreme_mask

            # Correct the non-extreme, non-NaN subset via the fitted quantile mapping.
            if non_extreme_target_mask.any():
                simp_target = simp_values[non_extreme_target_mask]
                simp_target_da = xr.DataArray(
                    simp_target, dims="time",
                    coords={"time": np.arange(len(simp_target))}, name=col,
                )
                corrected_target = adjust(
                    method=method,
                    obs=obs_da,
                    simh=simh_fit_da,
                    simp=simp_target_da,
                    kind=kind,
                    n_quantiles=n_quantiles,
                )
                simp_corrected[non_extreme_target_mask] = to_1d_array(corrected_target, col)

            if extrapolate_extremes:
                extreme_mask = simp_extreme_mask & ~simp_nan_mask

                if extreme_mask.any():
                    # Rather than calling adjust() directly on out-of-range simp
                    # values (relying on cmethods' own extrapolation), fit a
                    # straight line through the correction function evaluated at
                    # the top edge of the fitted (non-extreme) simh range, and
                    # extrapolate that line linearly beyond simh_threshold.
                    sorted_simh_fit = np.unique(np.sort(simh_fit))

                    if len(sorted_simh_fit) >= 2:
                        edge_simh = sorted_simh_fit[-2:]
                    else:
                        # Degenerate case: only one distinct fitted value. Fall
                        # back to a small synthetic offset so we still get two
                        # distinct points to define a line.
                        span = simh_fit.max() - simh_fit.min() if len(simh_fit) > 1 else 1.0
                        edge_simh = np.array([sorted_simh_fit[-1] - max(span, 1e-6) * 0.01,
                                               sorted_simh_fit[-1]])

                    edge_da = xr.DataArray(
                        edge_simh, dims="time",
                        coords={"time": np.arange(len(edge_simh))}, name=col,
                    )
                    edge_corrected = to_1d_array(
                        adjust(
                            method=method,
                            obs=obs_da,
                            simh=simh_fit_da,
                            simp=edge_da,
                            kind=kind,
                            n_quantiles=n_quantiles,
                        ),
                        col,
                    )

                    denom = edge_simh[1] - edge_simh[0]
                    if np.isclose(denom, 0.0):
                        # Can't determine a slope; fall back to identity for
                        # extremes rather than dividing by ~zero.
                        simp_corrected[extreme_mask] = simp_values[extreme_mask]
                    else:
                        # slope = (edge_corrected[1] - edge_corrected[0]) / denom
                        intercept = edge_corrected[1] - edge_simh[1]
                        # TODO: choose between two
                        # extrapolated = intercept + slope * simp_values[extreme_mask]
                        # simp_corrected[extreme_mask] = np.minimum(extrapolated, extrapolate_cutoff)
                        simp_corrected[extreme_mask] = intercept + simp_values[extreme_mask]
            # Non-targeted entries when extrapolate_extremes is False (extreme,
            # or originally-NaN) are left as their original value.

            return simp_corrected

        # Correct the held-out (validation) forecasts.
        X_val_corrected[col] = apply_correction(X_val[col].values)

        # Correct the training forecasts against themselves, so the model
        # trains on the same corrected distribution it will see at inference.
        X_train_corrected[col] = apply_correction(X_train[col].values)

    return X_train_corrected, X_val_corrected

