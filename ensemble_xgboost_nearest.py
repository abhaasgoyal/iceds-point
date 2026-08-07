import xarray as xr
import argparse
from sklearn.model_selection import train_test_split, LeaveOneGroupOut, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import xgboost as xgb
from xgboost import plot_importance
from xarray import DataArray, Dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import rioxarray 
from shapely.geometry import Polygon, Point
from feature_engineering import FeatureEngineering
from concurrent.futures import ProcessPoolExecutor
import csv
from cmethods import adjust


parser = argparse.ArgumentParser(
                    prog='ensemble_xgboost',
                    description='Ensemble XGBoost')

parser.add_argument('-r', '--region', default=1, type=int, choices=list(range(3)))
parser.add_argument('-l', '--lead_time', default=1, type=int, choices=list(range(1, 7)))
parser.add_argument('-f', '--feature', default="cbrt", type=str, choices=["cbrt", "log"])
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
    feature_dataset.select_lead_time(args.lead_time)
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

    if args.feature == "log":
        dataset["pr"] = np.log(dataset["pr"] + 1)
    elif args.feature == "cbrt":
        dataset["pr"] = np.cbrt(dataset["pr"])
    else:
        raise AssertionError()


    dataset = dataset[dataset["month"].isin([10, 11, 12, 1, 2, 3])]
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

    if is_yearly_split:
        logo = LeaveOneGroupOut()
        groups = dataset["year"]
    
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror')

        y_pred = np.empty(0, dtype=float)

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

            # TODO: Need to convert to NetCDF back and forth

            # adjust(
            #     method="quantile_mapping",
            #     obs = y_train,
            #     simh = X_train,
            #     simp= X_val,
            #     kind="+",
            #     n_quantiles=80,
            # )

            xgb_model = get_xgboost_model()
            clf_xgb_model = GridSearchCV(xgb_model, param_grid)

            clf_xgb_model.fit(X_train.drop(columns=["year", "is_centre"]), y_train) # Uses 5-fold cross validation

            y_val_pred = clf_xgb_model.predict(X_val.drop(columns=["year", "is_centre"]))
            # if silo_radius > 0.0:
            #     y_val_pred = y_val_pred[4::9]  # Select only middle points
            #     print(len(y_val_pred))

            print(clf_xgb_model.best_estimator_.feature_importances_)
            y_pred = np.append(y_pred, y_val_pred)
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
        testing = X.copy()
        testing["XGBoost"] = y_pred
        testing["obs_pr"] = y
        if silo_radius > 0:
            # idx = 8 * (round(silo_radius / 0.05)) + 1
            idx = points # len(points[0])
            print(idx)
            testing = testing[X["is_centre"] == 1]
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
elif range_id == 1:
    lat_range = (-29, -25)
    lon_range = (150, 154)
elif range_id == 2:
    lat_range = (-20, -16)
    lon_range = (143, 147) 

point_arg_list = []
for lat in range(*lat_range):
    for lon in range(*lon_range):
        point_arg_list.append((np.float64(lat), np.float64(lon)))
        # print(f"Using {lat}, {lon}")
        # run_single_point_analysis((np.float64(lat), np.float64(lon)))

with ProcessPoolExecutor() as executor:
    results = executor.map(run_single_point_analysis, point_arg_list)

rows = list(results)


headers = ["Point", "XGBoost_mae", "ecmwf_mae", "ukmo_mae", "access_mae", "XGBoost_rmse", "ecmwf_rmse", "ukmo_rmse", "access_rmse"]
with open(f"/scratch/tm70/ag9761/iceds_datasets/output_region{range_id}_lead{args.lead_time}_feature{args.feature}.csv", mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(headers)   # Writes a single row (header)
    writer.writerows(rows)     # Writes multiple data rows at once