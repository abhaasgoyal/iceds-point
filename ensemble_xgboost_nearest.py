import argparse
import csv
import re
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import seaborn as sns
import xarray as xr
import xgboost as xgb
from cmethods import adjust
from shapely.geometry import Point, Polygon
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from xarray import DataArray, Dataset
from xgboost import plot_importance

from feature_engineering import FeatureEngineering

parser = argparse.ArgumentParser(
                    prog='ensemble_xgboost',
                    description='Ensemble XGBoost')

parser.add_argument('-r', '--region', default=1, type=int, choices=list(range(4)))
parser.add_argument('-l', '--lead_time', default=1, type=int, choices=list(range(1, 7)))
parser.add_argument('-f', '--feature', default="cbrt", type=str, choices=["cbrt", "log"])
parser.add_argument('-s', '--season', default="wet", type=str, choices=["wet", "dry"])


# Bias correction options
parser.add_argument('--bc_exclude_percentile', default=95.0, type=float,
                     help="Percentile (on obs and simh separately) above which values "
                          "are treated as 'extreme' and excluded from the quantile "
                          "mapping fit.")
parser.add_argument('--bc_extrapolate_extremes', action='store_true',
                     help="If set, the mapping fit on non-extreme values is still "
                          "applied to (extrapolated onto) extreme values. If not set "
                          "(default), extreme values are left uncorrected (identity).")

args = parser.parse_args()

rows = []
range_id = args.region
print(f"Region: {range_id}")

# Config options
is_single_point = True

model_radius = 0.25
silo_radius = 0.25
n_points_inner_circle = 6
radius_inner_circle = 0.6
n_points_outer_circle = 6
radius_outer_circle = 1.039



def run_single_point_analysis(grid_point):
    # Datasets
    silo = xr.open_dataset(f"/scratch/tm70/ag9761/iceds_datasets/silo_monthly_rain_iceds_{range_id}_processed.nc")
    gcms = xr.open_dataset(f"/scratch/tm70/ag9761/iceds_datasets/preprocessed_dataset_{range_id}_new.nc")

    # Initial Analysis
    def single_point_data_analysis():
 
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (15,5),sharey=True)
        fig.suptitle("Comparison of Observed Precipitation values vs GCMs for region (150.0, -29.0)")
        for idx, model in enumerate([gcms.sel(model="ecmwf", lead=0), gcms.sel(model="access", lead=0), gcms.sel(model="ukmo", lead=0)]):

            scatter_plot_data = xr.merge([model, silo.rename_vars({"pr" : "obs_pr"})], compat="override")
            ax = sns.regplot(scatter_plot_data.to_dataframe(), x="pr", y="obs_pr", ax=axes[idx])
            ax.axline((0, 0), slope=1, color='k', ls='--')
            plt.legend(loc='upper left', labels=[None, "Regression Line Fit", None, "Slope 1"])
            ax.grid(True, axis='both', ls=':')
            ax.set_title(model.model.values)

    def construct_circle(origin, n_points, radius, base_angle=0.0, include_origin=False):

        def get_circular_point(angle):
            print(angle)
            return (round(origin[0] + radius * xr.ufuncs.cos(angle), 2), round(origin[1] + radius * xr.ufuncs.sin(angle), 2))
    
        coords = [get_circular_point(base_angle + 2 * i * np.pi / n_points_inner_circle) for i in range(n_points)]
        if include_origin:
            coords.insert(len(coords) // 2, origin)

        return tuple(map(list, zip(*coords)))


    if is_single_point:

        # gcms = gcms.sel(latitude=slice(grid_point[0] - model_radius, grid_point[0] + model_radius), longitude=slice(grid_point[1] - model_radius, grid_point[1] + model_radius))

        ## Inner circle
        points_1 = construct_circle(grid_point, n_points_inner_circle, radius_inner_circle, include_origin=True)
        points_2 = construct_circle(grid_point, n_points_outer_circle, radius_outer_circle, base_angle=np.pi / 6)
        points = (points_1[0] + points_2[0], points_1[1] + points_2[1])
        # points = points_1
        print(points[0])
        print(points[1])
        print(len(points[0]))
        # lat_da = xr.DataArray(points[0], dims="point")
        # lon_da = xr.DataArray(points[1], dims="point")

        def select_circular_points(dataset, points):
            lat_da = xr.DataArray(points[0], dims="point")
            lon_da = xr.DataArray(points[1], dims="point")
            dataset = dataset.sel(latitude=lat_da, longitude=lon_da, method="nearest")
            print(dataset.latitude.values)
            print(dataset.longitude.values)
            dataset = dataset.assign_coords(point=pd.MultiIndex.from_arrays([dataset.latitude.values, dataset.longitude.values], names=["latitude", "longitude"])).unstack("point")
            dataset = dataset.sortby(["latitude", "longitude"])
            return dataset

        def dataset_within_radius(dataset, grid_point, radius, max_radius=0.40, cutoff_points=81):

            # Incrementally increase radius until hit minimum number of poitns
            while radius <= max_radius:
        
                dist = np.hypot(dataset.latitude - grid_point[0], dataset.longitude - grid_point[1])
                mask = xr.ufuncs.bitwise_or(dist <= radius, np.isclose(dist, radius))

                selected = dataset.where(mask, drop=True)

                selected_stacked = selected.isel(time=0)["pr"].stack(x=['latitude','longitude'])
                points = len(selected_stacked[selected_stacked.notnull()])

                if points >= cutoff_points:
                    break

                radius += 0.05

            if points < cutoff_points:
                raise AssertionError(f"n_points: {points} Cutoff Points: {cutoff_points} Location: {grid_point}")
                
            return selected, points


        gcms["latitude"] = gcms["latitude"].round(decimals=2)
        gcms["longitude"] = gcms["longitude"].round(decimals=2)
        silo["longitude"] = silo["longitude"].round(decimals=2)
        silo["latitude"] = silo["latitude"].round(decimals=2)
        
        # Option 1

        cutoff_points = 81
        silo, points = dataset_within_radius(silo, grid_point, model_radius, cutoff_points)
        # gcms = dataset_within_radius(gcms, grid_point, model_radius)

        mask = silo["pr"].notnull()
        gcms = gcms.where(mask, drop=True)

        


        # Option 2
        # silo = select_circular_points(silo, points)
        # gcms = select_circular_points(gcms, points)

    # single_point_data_analysis()
    # plt.show()
    # plt.clf()

    silo_stacked = silo.isel(time=0)["pr"].stack(x=['latitude','longitude'])
    silo_df = silo_stacked[silo_stacked.notnull()].to_dataframe().reset_index()
    # plt.scatter(silo_df["latitude"], silo_df["longitude"])
    # plt.show()
    # plt.clf()


    feature_dataset = FeatureEngineering(gcms, lead_months=args.lead_time) 
    # feature_dataset.enable_wet_season_feature()
    # feature_dataset.enable_climate_classification()
    # feature_dataset.select_lead_time(args.lead_time)
    feature_dataset.create_lead_time_features(["pr"])
    # feature_dataset.add_obs_window_time(period=1)



    dataset = xr.merge([feature_dataset.ds, silo], join="outer")
    dataset = dataset.sel(time=slice("1993-01-01", "2016-12-31"))

    dataset = dataset.to_dataframe().sort_values(by=['time', "latitude", "longitude"])

    dataset["is_centre"] = np.where((np.isclose(dataset.index.get_level_values("latitude"), grid_point[0])) & (np.isclose(dataset.index.get_level_values("longitude"), grid_point[1])), 1, 0)
    assert dataset["is_centre"].sum() == 288

    # Rounded Index to integer (upto 0.5 degree)
    
    if silo_radius > 0:
        rounded_index = pd.MultiIndex.from_tuples(
            [(round(t[0]), round(t[1]), t[2]) for t in dataset.index],
            names=dataset.index.names
        )
        dataset.index = rounded_index
    dataset = dataset.dropna(subset=["pr"]) # Could be because points selected before outer merge could be Nan
    dataset = dataset.reset_index().drop(columns=["time"])

    if args.season == "wet":
        dataset = dataset[dataset["month"].isin([10, 11, 12, 1, 2, 3])]
    else:
        dataset = dataset[~dataset["month"].isin([10, 11, 12, 1, 2, 3])]
        # TODO: Check
        # Treat trace precipitation as effectively zero (< 0.1 mm/day)
        dataset["pr"] = np.where(dataset["pr"] < 0.1, 0.0, dataset["pr"])

    if args.feature == "log":
        dataset["pr"] = np.log(dataset["pr"] + 1)
    elif args.feature == "cbrt":
        dataset["pr"] = np.cbrt(dataset["pr"])
    else:
        raise AssertionError()




    # dataset = pd.get_dummies(dataset, columns=["month"], dtype=int)
    dataset = dataset.drop(columns=["month", "latitude", "longitude"]) 

    print(dataset)


    is_yearly_split = True
    X = dataset.drop(columns=["pr"])
    y = dataset["pr"]

    def get_xgboost_model():
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric= 'mae', max_depth=5, eta=0.1, 
                                subsample=0.7, colsample_bytree=0.7)

        return xgb_model

    def xgboost_train(X_train, y_train):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'max_depth': 5,
            'eta': 0.1,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'verbosity': 0
        }

        xgb_model = get_xgboost_model()

        xgb_model.fit(X_train, y_train)

        return xgb_model


    def xgboost_predict(xgb_model):
        return xgb_model.predict(X_test)

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
            points are formed by evaluating the fitted mapping at the top two
            distinct fitted simh values; a line is fit through those two
            (simh, corrected) points and extended linearly beyond simh_threshold.

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

 
    if is_yearly_split:
        logo = LeaveOneGroupOut()
        groups = dataset["year"]
    
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror')

        y_pred = np.empty(0, dtype=float)
        X_val_accum = []

        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [2],
            'min_child_weight': [1, 3],
            'subsample': [0.2, 0.3, 0.4, 0.7],
            'colsample_bytree': [0.7], 
            'reg_alpha': [1.0],          # L1 regularization
            'reg_lambda': [5.0],         # L2 regularization (increase this)
            'eta': [0.1],
            'n_estimators': [5, 10, 20, 30, 50]
        }

        for train_idx, val_idx in logo.split(X, y, groups):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # y_train is on the cbrt/log-transformed scale (applied to "pr"
            # earlier), while the GCM forecast columns in X_train/X_val are
            # still raw rainfall values. Invert the transform so obs and
            # simh/simp are on the same scale before quantile mapping.
            if args.feature == "log":
                y_train_raw = np.exp(y_train) - 1
            elif args.feature == "cbrt":
                y_train_raw = np.power(y_train, 3)
            else:
                raise AssertionError()

            X_train_ecmwf_before = X_train["ecmwf_0_pr"].copy()
            # X_train["ecmwf_0_pr"].plot.hist(
            #     density=True, 
            #     cumulative=True, 
            #     histtype='step', 
            #     bins=80, 
            #     title="Approximated CDF (before)"
            # )
            # plt.show()

            # print("YTRAIN")
            # print(y_train_raw.describe())
            # print("XTRAIN")
            # print(X_train["ecmwf_0_pr"].describe())

            # X_train, X_val = bias_correct_with_cmethods(
            #     X_train, X_val, y_train_raw,
            #     method="quantile_mapping", kind="+", n_quantiles=80,
            # )
            X_train, X_val = bias_correct_with_cmethods(
                X_train, X_val, y_train_raw,
                method="quantile_mapping", kind="+", n_quantiles=80,
                exclude_percentile=args.bc_exclude_percentile,
                extrapolate_extremes=args.bc_extrapolate_extremes,
            )
            # #-------
            # Comparison of bias-correction behaviour across several
            # exclude_percentile values, holding extrapolate_extremes fixed
            # at whatever was passed on the CLI. This is purely diagnostic -
            # the actual X_train/X_val used for training above already come
            # from the single bias_correct_with_cmethods call using
            # args.bc_exclude_percentile.
            bc_percentiles_to_compare = [85.0, 90.0, 95.0, 100.0]
            bc_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

            X_train_pctl_results = {}
            for pctl in bc_percentiles_to_compare:
                X_train_pctl, _ = bias_correct_with_cmethods(
                    X_train, X_val, y_train_raw,
                    method="quantile_mapping", kind="+", n_quantiles=80,
                    exclude_percentile=pctl,
                    extrapolate_extremes=args.bc_extrapolate_extremes,
                )
                X_train_pctl_results[pctl] = X_train_pctl

            # Bar chart: mean bias before vs. after, grouped by percentile.
            # bias_before = (X_train_ecmwf_before - y_train_raw.values).mean()
            # bias_after_by_pctl = {
            #     f"after (p={pctl:.0f})": (X_train_pctl_results[pctl]["ecmwf_0_pr"] - y_train_raw.values).mean()
            #     for pctl in bc_percentiles_to_compare
            # }
            # bias_series = pd.Series({"before": bias_before, **bias_after_by_pctl})
            # ax_bar = bias_series.plot.bar(
            #     title="ECMWF Mean Bias Before vs After (by Exclude Percentile)",
            #     color=["gray"] + bc_colors,
            # )
            # ax_bar.axhline(0, color='k', lw=1)
            # ax_bar.set_ylabel("Mean Bias")
            # plt.tight_layout()
            # plt.show()

            # # Scatter: before vs. after correction, one color per percentile.
            # _, ax_scatter = plt.subplots(figsize=(7, 7))
            # ax_scatter.scatter(
            #     X_train_ecmwf_before, X_train_ecmwf_before,
            #     s=10, alpha=0.6, color="gray", label="uncorrected",
            # )
            # for pctl, color in zip(bc_percentiles_to_compare, bc_colors):
            #     ax_scatter.scatter(
            #         X_train_ecmwf_before, X_train_pctl_results[pctl]["ecmwf_0_pr"],
            #         s=10, alpha=0.6, color=color, label=f"p={pctl:.0f}",
            #     )
            # lo = min(ax_scatter.get_xlim()[0], ax_scatter.get_ylim()[0])
            # hi = max(ax_scatter.get_xlim()[1], ax_scatter.get_ylim()[1])
            # ax_scatter.plot([lo, hi], [lo, hi], 'k--', lw=1, label="1:1")
            # ax_scatter.set_xlabel("Before")
            # ax_scatter.set_ylabel("After")
            # ax_scatter.set_title("ECMWF Before vs After Correction (by Exclude Percentile)")
            # ax_scatter.legend()
            # plt.tight_layout()
            # plt.show()
            # #----------------
            # pd.Series({"before": (X_train_ecmwf_before - y_train_raw.values).mean(), "after": (X_train["ecmwf_0_pr"] - y_train_raw.values).mean()}).plot.bar(title="ECMWF Mean Bias Before vs After"); plt.axhline(0, color='k', lw=1); plt.show()

            # plt.scatter(X_train_ecmwf_before, X_train["ecmwf_0_pr"])

            # plt.show()

            # X_train["ecmwf_0_pr"].plot.hist(
            #     density=True, 
            #     cumulative=True, 
            #     histtype='step', 
            #     bins=80, 
            #     title="Approximated CDF (after)"
            # )
            # plt.show()


            xgb_model = get_xgboost_model()
            clf_xgb_model = GridSearchCV(xgb_model, param_grid)

            clf_xgb_model.fit(X_train.drop(columns=["year", "is_centre"]), y_train) # Uses 5-fold cross validation

            y_val_pred = clf_xgb_model.predict(X_val.drop(columns=["year", "is_centre"]))
            # if silo_radius > 0.0:
            #     y_val_pred = y_val_pred[4::9]  # Select only middle points
            #     print(len(y_val_pred))

            print(clf_xgb_model.best_estimator_.feature_importances_)
            y_pred = np.append(y_pred, y_val_pred)
            X_val_accum.append(X_val)
            importance = clf_xgb_model.best_params_

        
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.05,
            shuffle=False
        )
        print(X_test)
        xgb_model = xgboost_train(X_train, y_train)

        y_pred = xgboost_predict(xgb_model)


    ## Evaluation

    def single_point_data_analysis(data):
 
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (10,10),sharey=True)
        fig.suptitle("Testing of Observed Precipitation values vs GCMs for region (150.0, -29.0)")
        for idx, model in enumerate(["ecmwf_0_pr", "ukmo_0_pr", "access_0_pr", "XGBoost"]):
        # for idx, model in enumerate(["ecmwf_0_pr", "access_0_pr", "XGBoost"]):
            i = idx // 2
            j = idx % 2

            ax = sns.regplot(data, x=model, y="obs_pr", ax=axes[i, j])
            ax.set(xlim=(0, 10))
            ax.axline((0, 0), slope=1, color='k', ls='--')
            plt.legend(loc='upper left', labels=[None, "Regression Line Fit", None, "Slope 1"])
            ax.grid(True, axis='both', ls=':')
            ax.set_title(model)
        plt.show()

    if args.feature == "log":
        y_pred = np.exp(y_pred) - 1
        y = np.exp(y) - 1
    elif args.feature == "cbrt":
        y_pred = np.power(y_pred, 3)
        y = np.power(y, 3)

    if is_yearly_split:
        X_val_all = pd.concat(X_val_accum)
        print(X_val_all)
        testing = X_val_all.copy()
        testing["XGBoost"] = y_pred
        testing["obs_pr"] = y
        if silo_radius > 0:
            # idx = 8 * (round(silo_radius / 0.05)) + 1
            idx = points # len(points[0])
            print(idx)
            testing = testing[testing["is_centre"] == 1]
            # testing = testing[(idx//2)::idx]  # Select only middle points

            print(testing)
    else:
        testing = X_test.copy()
        testing["XGBoost"] = y_pred
        testing["obs_pr"] = y_test

    # single_point_data_analysis(testing)


    testing_temp = testing # .drop(testing.index[-4])
    testing_temp = testing_temp.iloc[(args.lead_time - 1):]
    ukmo_mask = ~np.isnan(testing["obs_pr"]) & ~np.isnan(testing[f"ukmo_{args.lead_time - 1}_pr"])

    model_mae = {
        "XGBoost" : mean_absolute_error(testing_temp["obs_pr"], testing_temp["XGBoost"]),
        "ecmwf" : mean_absolute_error(testing_temp["obs_pr"], testing_temp[f"ecmwf_{args.lead_time - 1}_pr"]),
        "ukmo" : mean_absolute_error(testing["obs_pr"][ukmo_mask], testing[f"ukmo_{args.lead_time - 1}_pr"][ukmo_mask]),
        "access" : mean_absolute_error(testing_temp["obs_pr"], testing_temp[f"access_{args.lead_time - 1}_pr"])
    }

    for model, mae in model_mae.items():
        print(f"{model} MAE: {mae}")


    model_rmse = {
        "XGBoost" : root_mean_squared_error(testing_temp["obs_pr"], testing_temp["XGBoost"]),
        "ecmwf" : root_mean_squared_error(testing_temp["obs_pr"], testing_temp[f"ecmwf_{args.lead_time - 1}_pr"]),
        "ukmo" : root_mean_squared_error(testing["obs_pr"][ukmo_mask], testing[f"ukmo_{args.lead_time - 1}_pr"][ukmo_mask]),
        "access" : root_mean_squared_error(testing_temp["obs_pr"], testing_temp[f"access_{args.lead_time - 1}_pr"])
    }

    for model, rmse in model_rmse.items():
        print(f"{model} RMSE: {rmse}")

    return [grid_point, model_mae["XGBoost"], model_mae["ecmwf"], model_mae["ukmo"], model_mae["access"], model_rmse["XGBoost"], model_rmse["ecmwf"], model_rmse["ukmo"], model_rmse["access"]]


if range_id == 0:
    lat_range = (-25, -21)
    lon_range = (142, 146)
# elif range_id == 1:
#     lat_range = (-29, -25)
#     lon_range = (150, 154)
elif range_id == 1:
    lat_range = (-29, -28)
    lon_range = (150, 151)
# elif range_id == 2:
#     lat_range = (-29, -28)
#     lon_range = (150, 151) 
elif range_id == 2:
    lat_range = (-17, -16)
    lon_range = (143, 147) 
elif range_id == 3:
    lat_range = (-29, -9)
    lon_range = (138, 154) 

def is_valid_silo_point(lat, lon, tolerance=0.02, min_valid_fraction=0.8):

    silo_mask_ds = xr.open_dataset("/scratch/tm70/ag9761/iceds_datasets/silo_monthly_rain_iceds_3_processed.nc")
    silo_mask_ds["latitude"] = silo_mask_ds["latitude"].round(decimals=2)
    silo_mask_ds["longitude"] = silo_mask_ds["longitude"].round(decimals=2)
    """
    True if (lat, lon) has a nearby SILO grid cell with real (non-NaN) data
    for a sufficient fraction of the time series - i.e. it's a genuine land
    point within SILO's coverage, not ocean or a gap in the grid.

    tolerance: max distance (degrees) to the nearest SILO grid cell before
    we consider it "not on the grid at all" (SILO is on a 0.05 deg grid,
    so 0.13 is generous but still rejects points far outside coverage).
    min_valid_fraction: fraction of the time series that must be non-NaN
    at the nearest cell for the point to count as valid land.
    """
    nearest = silo_mask_ds["pr"].sel(latitude=lat, longitude=lon, method="nearest")

    actual_lat = float(nearest["latitude"].values)
    actual_lon = float(nearest["longitude"].values)
    if abs(actual_lat - lat) > tolerance or abs(actual_lon - lon) > tolerance:
        return False

    valid_fraction = float(nearest.notnull().mean())
    return valid_fraction >= min_valid_fraction

point_arg_list = []
for lat in range(*lat_range):
    for lon in range(*lon_range):
        # if lat == -17 and lon == 146:
        #     continue

        if not is_valid_silo_point(np.float64(lat), np.float64(lon)):
            print(f"Skipping ({lat}, {lon}): not a valid SILO land point")
            continue
        point_arg_list.append((np.float64(lat), np.float64(lon)))
        # print(f"Using {lat}, {lon}")
        # run_single_point_analysis((np.float64(lat), np.float64(lon)))

with ProcessPoolExecutor() as executor:
    results = executor.map(run_single_point_analysis, point_arg_list)

rows = list(results)


headers = ["Point", "XGBoost_mae", "ecmwf_mae", "ukmo_mae", "access_mae", "XGBoost_rmse", "ecmwf_rmse", "ukmo_rmse", "access_rmse"]
with open(f"/scratch/tm70/ag9761/iceds_datasets/output_region{range_id}_lead{args.lead_time}_feature{args.feature}_perc{args.bc_exclude_percentile}.csv", mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(headers)   # Writes a single row (header)
    writer.writerows(rows)     # Writes multiple data rows at once