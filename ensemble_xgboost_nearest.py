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

from bias_correction import bias_correct_gcm_features, bias_correct_predictions
from feature_engineering import FeatureEngineering, inverse_transform, prepare_dataset

parser = argparse.ArgumentParser(
                    prog='ensemble_xgboost',
                    description='Ensemble XGBoost')

parser.add_argument('-r', '--region', default=1, type=int, choices=list(range(4)))
parser.add_argument('-l', '--lead_time', default=1, type=int, choices=list(range(1, 7)))
parser.add_argument('-f', '--feature', default="cbrt", type=str, choices=["cbrt", "log"])
parser.add_argument('-s', '--season', default="wet", type=str, choices=["wet", "dry"])
parser.add_argument('--forecast', action='store_true',
                     help="If set, run in forecast mode: use the forecast-extended "
                          "datasets, skip the 2025/2026 year filter, evaluate with a "
                          "single train/val split on 2025-2026, and write output under "
                          "the 'forecast' output directory. If not set (default), run "
                          "in hindcast mode: use the hindcast datasets, drop 2025/2026 "
                          "from training, evaluate with leave-one-year-out cross "
                          "validation (with bias correction applied to predictions "
                          "too), and write output under the 'hindcast' output "
                          "directory.")


# Bias correction options
parser.add_argument('--no_bias_correction', action='store_true',
                     help="If set, skip all bias correction: neither the GCM "
                          "forecast features nor the XGBoost predictions are "
                          "quantile-mapped.")
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
if args.forecast:
    print("In forecast mode:")
else:
    print("In hindcast mode:")
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
    silo = xr.open_dataset("/scratch/tm70/ag9761/iceds_datasets/silo_monthly_rain_iceds_3_processed_forecast.nc")
    if args.forecast:
        gcms = xr.open_mfdataset(["/scratch/tm70/ag9761/iceds_datasets/preprocessed_dataset_3_new.nc", 
                                  "/scratch/tm70/ag9761/iceds_datasets/preprocessed_dataset_3_new_forecast.nc"],
                                  chunks="auto",
                                  compat="override",
                                  coords="minimal")
    else:
        gcms = xr.open_dataset("/scratch/tm70/ag9761/iceds_datasets/preprocessed_dataset_3_new.nc")


    def construct_circle(origin, n_points, radius, base_angle=0.0, include_origin=False):

        def get_circular_point(angle):
            print(angle)
            return (round(origin[0] + radius * xr.ufuncs.cos(angle), 2), round(origin[1] + radius * xr.ufuncs.sin(angle), 2))

        coords = [get_circular_point(base_angle + 2 * i * np.pi / n_points_inner_circle) for i in range(n_points)]
        if include_origin:
            coords.insert(len(coords) // 2, origin)

        return tuple(map(list, zip(*coords)))


    if is_single_point:

        ## Inner circle
        points_1 = construct_circle(grid_point, n_points_inner_circle, radius_inner_circle, include_origin=True)
        points_2 = construct_circle(grid_point, n_points_outer_circle, radius_outer_circle, base_angle=np.pi / 6)
        points = (points_1[0] + points_2[0], points_1[1] + points_2[1])
        print(points[0])
        print(points[1])
        print(len(points[0]))

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

        cutoff_points = 81
        # TODO: Fix more optimally
        silo, points = dataset_within_radius(silo, grid_point, model_radius, cutoff_points)
        # gcms = dataset_within_radius(gcms, grid_point, model_radius)

        mask = silo["pr"].notnull()
        gcms = gcms.where(mask, drop=True)

    # single_point_data_analysis()
    # plt.show()
    # plt.clf()

    silo_stacked = silo.isel(time=0)["pr"].stack(x=['latitude','longitude'])
    silo_df = silo_stacked[silo_stacked.notnull()].to_dataframe().reset_index()
    # plt.scatter(silo_df["latitude"], silo_df["longitude"])
    # plt.show()
    # plt.clf()


    feature_dataset = FeatureEngineering(gcms, lead_months=args.lead_time)
    feature_dataset.enable_seasonality_feature()
    # feature_dataset.enable_wet_season_feature()
    # feature_dataset.enable_climate_classification()
    feature_dataset.select_lead_time(args.lead_time)
    feature_dataset.create_lead_time_features(["pr"])
    # feature_dataset.add_obs_window_time(period=1)

    dataset = prepare_dataset(feature_dataset, silo, grid_point, silo_radius, args, is_forecast=args.forecast)

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

    if is_yearly_split:

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

        if args.forecast:

            val_mask = dataset["year"].isin([2025, 2026])
            X_train, X_val = X[~val_mask], X[val_mask]
            y_train, y_val = y[~val_mask], y[val_mask]
            print(X_val)

            xgb_model = xgb.XGBRegressor(objective='reg:squarederror')

            # y_train is on the cbrt/log-transformed scale (applied to "pr"
            # earlier), while the GCM forecast columns in X_train/X_val are
            # still raw rainfall values. Invert the transform so obs and
            # simh/simp are on the same scale before quantile mapping.

            # X_train["ecmwf_0_pr"].plot.hist(
            #     density=True,
            #     cumulative=True,
            #     histtype='step',
            #     bins=80,
            #     title="Approximated CDF (before)"
            # )
            # plt.show()

            if not args.no_bias_correction:
                print("Doing bias correction")
                X_train, X_val = bias_correct_gcm_features(X_train, X_val, y_train, args)


            xgb_model = get_xgboost_model()
            clf_xgb_model = GridSearchCV(xgb_model, param_grid)

            clf_xgb_model.fit(X_train.drop(columns=["year", "is_centre"]), y_train) # Uses 5-fold cross validation

            y_val_pred = clf_xgb_model.predict(X_val.drop(columns=["year", "is_centre"]))
            y_train_pred = clf_xgb_model.predict(X_train.drop(columns=["year", "is_centre"]))


            # Bias correct on XGBoost predictions
            if not args.no_bias_correction:
                y_val_pred = bias_correct_predictions(y_train_pred, y_val_pred, y_train, args)


            y_pred = y_val_pred
            # if silo_radius > 0.0:
            #     y_pred = y_pred[4::9]  # Select only middle points
            #     print(len(y_pred))

            print(clf_xgb_model.best_estimator_.feature_importances_)
            print(X_val.describe())
            importance = clf_xgb_model.best_params_

        else:
            logo = LeaveOneGroupOut()
            groups = dataset["year"]

            xgb_model = xgb.XGBRegressor(objective='reg:squarederror')

            y_pred = np.empty(0, dtype=float)
            X_val_accum = []

            for train_idx, val_idx in logo.split(X, y, groups):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # y_train is on the cbrt/log-transformed scale (applied to "pr"
                # earlier), while the GCM forecast columns in X_train/X_val are
                # still raw rainfall values. Invert the transform so obs and
                # simh/simp are on the same scale before quantile mapping.

                # X_train["ecmwf_0_pr"].plot.hist(
                #     density=True,
                #     cumulative=True,
                #     histtype='step',
                #     bins=80,
                #     title="Approximated CDF (before)"
                # )
                # plt.show()

                if not args.no_bias_correction:
                    print("Doing bias correction")
                    X_train, X_val = bias_correct_gcm_features(X_train, X_val, y_train, args)

                xgb_model = get_xgboost_model()
                clf_xgb_model = GridSearchCV(xgb_model, param_grid)

                clf_xgb_model.fit(X_train.drop(columns=["year", "is_centre"]), y_train) # Uses 5-fold cross validation

                y_val_pred = clf_xgb_model.predict(X_val.drop(columns=["year", "is_centre"]))
                y_train_pred = clf_xgb_model.predict(X_train.drop(columns=["year", "is_centre"]))


                # Bias correct on XGBoost predictions
                if not args.no_bias_correction:
                    y_val_pred = bias_correct_predictions(y_train_pred, y_val_pred, y_train, args)


                # if silo_radius > 0.0:
                #     y_val_pred = y_val_pred[4::9]  # Select only middle points
                #     print(len(y_val_pred))

                print(clf_xgb_model.best_estimator_.feature_importances_)
                y_pred = np.append(y_pred, y_val_pred)
                print(X_val.describe())
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
        fig.suptitle("Lead 1: Testing of Observed Precipitation values vs GCMs for region (150.0, -29.0)")
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

    y_pred = inverse_transform(y_pred, args.feature)
    y = inverse_transform(y, args.feature)

    if is_yearly_split:
        if args.forecast:
            testing = X_val.copy()
        else:
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


    testing_temp = testing
    if not args.forecast:
        # In hindcast data, this is done since the models start predictions from Jan 1993
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

    # Bias = mean(forecast - observed): positive means the model
    # over-predicts on average, negative means it under-predicts.
    model_bias = {
        "XGBoost" : np.mean(testing_temp["XGBoost"] - testing_temp["obs_pr"]),
        "ecmwf" : np.mean(testing_temp[f"ecmwf_{args.lead_time - 1}_pr"] - testing_temp["obs_pr"]),
        "ukmo" : np.mean(testing[f"ukmo_{args.lead_time - 1}_pr"][ukmo_mask] - testing["obs_pr"][ukmo_mask]),
        "access" : np.mean(testing_temp[f"access_{args.lead_time - 1}_pr"] - testing_temp["obs_pr"])
    }

    for model, bias in model_bias.items():
        print(f"{model} Bias: {bias}")

    return [grid_point, model_mae["XGBoost"], model_mae["ecmwf"], model_mae["ukmo"], model_mae["access"], model_rmse["XGBoost"], model_rmse["ecmwf"], model_rmse["ukmo"], model_rmse["access"], model_bias["XGBoost"], model_bias["ecmwf"], model_bias["ukmo"], model_bias["access"]]


if range_id == 0:
    lat_range = (-25, -21)
    lon_range = (142, 146)
elif range_id == 1:
    lat_range = (-29, -25)
    lon_range = (150, 154)
elif range_id == 2:
    lat_range = (-20, -16)
    lon_range = (143, 147)
# elif range_id == 3: # For debugging purposes
#     lat_range = (-29, -28)
#     lon_range = (150, 151)
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
        if not is_valid_silo_point(np.float64(lat), np.float64(lon)):
            print(f"Skipping ({lat}, {lon}): not a valid SILO land point")
            continue
        point_arg_list.append((np.float64(lat), np.float64(lon)))
        # print(f"Using {lat}, {lon}")
        # run_single_point_analysis((np.float64(lat), np.float64(lon)))

with ProcessPoolExecutor() as executor:
    results = executor.map(run_single_point_analysis, point_arg_list)

rows = list(results)

HOME_DIR = "/scratch/tm70/ag9761/iceds_datasets/temp/" # TODO: Change back

if args.no_bias_correction:
    print("no bias correction")

headers = ["Point", "XGBoost_mae", "ecmwf_mae", "ukmo_mae", "access_mae", "XGBoost_rmse", "ecmwf_rmse", "ukmo_rmse", "access_rmse", "XGBoost_bias", "ecmwf_bias", "ukmo_bias", "access_bias"]
cast_dir  = "forecast" if args.forecast else "hindcast"
bias_dir = "before" if args.no_bias_correction else "after"
output_dir = f"{HOME_DIR}/{cast_dir}/{args.season}/bias_analysis/{bias_dir}"

with open(f"{output_dir}/output_region{range_id}_lead{args.lead_time}_feature{args.feature}_perc{args.bc_exclude_percentile}.csv", mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(headers)   # Writes a single row (header)
    writer.writerows(rows)     # Writes multiple data rows at once