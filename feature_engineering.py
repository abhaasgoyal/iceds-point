import xarray as xr
from xarray import DataArray, Dataset
import numpy as np
import pandas as pd

class FeatureEngineering:

    def __init__(self, ds: Dataset, lead_months = 6, window_time = 0):
        self.window_time = window_time
        self.ds = ds.copy()
        self.ds = self.ds.isel(lead=slice(0, lead_months))
        self._process_dates()
    

    def _process_dates(self):
        self.ds["month"] = self.ds["time"].dt.month
        self.ds["year"] = self.ds["time"].dt.year
    
    def select_lead_time(self, lead_time: int):
        self.ds = self.ds.isel(lead=[lead_time - 1])

    def enable_climate_classification(self):
        # https://doi.org/10.1038/s41597-023-02549-6
        climate_class_ds = rioxarray.open_rasterio("koppen_geiger_0p00833333.tif")
        climate_class_ds = climate_class_ds.rename({"x": "longitude", "y": "latitude"})
        climate_class_ds = climate_class_ds.squeeze("band").drop_vars(["band", "spatial_ref"])

        classification = climate_class_ds.sel(latitude=self.ds.latitude, longitude=self.ds.longitude, method="nearest")
        classification = classification.reindex_like(self.ds, method="nearest")
        self.ds["classification"] = classification

    def enable_wet_season_feature(self):
        """Binary switch for dry/wet season"""
        wet_months = [10, 11, 12, 1, 2, 3]
        self.ds["is_wet_month"] = (self.ds["time"].dt.month.isin(wet_months)).astype(int)
    
    def create_lead_time_features(self, params: list[str]):
        for param in params:
            param_features = self._create_lead_time_features(param)
            self.ds = self.ds.drop_vars([param])
            self.ds = xr.merge([self.ds, param_features], join="outer")
        self.ds = self.ds.drop_vars(["lead", "model"])
        return self.ds

    def _create_lead_time_features(self, param: str) -> Dataset:
        """Have separate columns for each lead time forecast."""
        
        param_lead_time = (
            self.ds[param]
            .stack(lead_model=("lead", "model"))
            .groupby("lead_model")
            .map(lambda da: da.squeeze(drop=True))
            .to_dataset(dim="lead_model")
        )

        return param_lead_time.rename_vars({
            k: f"{k[1]}_{k[0]}_{param}"
            for k in param_lead_time.data_vars
        })
    
    def add_obs_window_time(self, period: int):
        for i in range(period):
            self.ds[f"window_{period - 1}"] = silo["pr"].shift(time=period)
    
    # TODO: Add climatology feature

def prepare_dataset(feature_dataset, silo, grid_point, silo_radius, args):
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
        # dataset["pr"] = np.where(dataset["pr"] < 0.1, 0.0, dataset["pr"])

    if args.feature == "log":
        dataset["pr"] = np.log(dataset["pr"] + 1)
    elif args.feature == "cbrt":
        dataset["pr"] = np.cbrt(dataset["pr"])
    else:
        raise AssertionError()

    # dataset = pd.get_dummies(dataset, columns=["month"], dtype=int)
    dataset = dataset.drop(columns=["month", "latitude", "longitude"]) 

    print(dataset)

    return dataset
