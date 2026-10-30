import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from xarray import DataArray, Dataset


class FeatureEngineering:

    def __init__(self, ds: Dataset, lead_months = 6, window_time = 0):
        self.window_time = window_time
        self.ds = ds.copy()
        self.ds = self.ds.isel(lead=slice(0, lead_months))
        self._process_dates()


    def _process_dates(self):
        self.ds["month"] = self.ds["time"].dt.month
        self.ds["year"] = self.ds["time"].dt.year

    def enable_seasonality_feature(self):
        """Cyclical month encoding."""
        angle = 2 * np.pi * (self.ds["month"] % 12) / 12
        self.ds["month_cos"] = np.cos(angle)
        self.ds["month_sin"] = np.sin(angle)


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

    def add_obs_window_time(self, period: int, silo):
        for i in range(period):
            self.ds[f"window_{period - 1}"] = silo["pr"].shift(time=period)

    # TODO: Add climatology feature


def _parse_forecast_month(args):
    """Return (year, month) for --forecast_month, or (None, None) if not set."""
    fm = getattr(args, "forecast_month", None)
    if fm is None:
        return None, None
    year, month = fm.split("-")
    return int(year), int(month)


def prepare_dataset(feature_dataset, silo, grid_point, silo_radius, args, is_forecast=False):
    dataset = xr.merge([feature_dataset.ds, silo], join="outer")
    n_forecast_months = 11 if args.lead_time == 6 else 12

    target_year, target_month = _parse_forecast_month(args)
    is_forecast_month = target_year is not None
    target_present = False

    time = dataset["time"]
    hindcast_mask = (time >= np.datetime64("1993-01-01")) & (time <= np.datetime64("2016-12-31"))

    if is_forecast_month:
        # Single-month forecast: training period (1993-2016) + the target month only.
        # The target month may have no SILO obs yet (e.g. after Oct 2026) - that's fine,
        # its rows are kept below even though "pr" is NaN.
        target_mask = (time.dt.year == target_year) & (time.dt.month == target_month)
        target_present = bool(target_mask.any())
        if not target_present:
            print(f"No GCM data for {args.forecast_month} at lead {args.lead_time} "
                  f"(init month after the last available indexing month)")
        dataset = dataset.sel(time=hindcast_mask | target_mask)

    elif is_forecast:
        lead_time_filter = "2025-10-01" if args.lead_time == 6 else "2025-09-01"
        mask = hindcast_mask | (
            (time >= np.datetime64(lead_time_filter)) & (time <= np.datetime64("2026-08-31"))
        )
        dataset = dataset.sel(time=mask) # xr.concat([hindcast_time, forecast_time], dim="time")
        # forecast period should be Oct-Aug (11 months) for lead 6, Sep-Aug (12 months) otherwise
        forecast_time = dataset["time"].where(dataset["time"] >= np.datetime64(lead_time_filter), drop=True)
        dataset_forecast_months = np.unique(forecast_time.values.astype("datetime64[M]")).size

        assert dataset_forecast_months == n_forecast_months, (
            f"expected {n_forecast_months} forecast months for lead {args.lead_time}, "
            f"got {dataset_forecast_months}"
        )
    else:
        dataset = dataset.sel(time=slice("1993-01-01", "2016-12-31"))

    dataset = dataset.to_dataframe().sort_values(by=['time', "latitude", "longitude"])

    dataset["is_centre"] = np.where((np.isclose(dataset.index.get_level_values("latitude"), grid_point[0])) & (np.isclose(dataset.index.get_level_values("longitude"), grid_point[1])), 1, 0)

    if is_forecast_month:
        assert dataset["is_centre"].sum() == 288 + int(target_present)
    elif is_forecast:
        assert dataset["is_centre"].sum() == 288 + n_forecast_months
    else:
        assert dataset["is_centre"].sum() == 288

    # Drop rows with no SILO obs: these are either outside SILO's coverage
    # (points selected before the outer merge could be NaN) or have no obs yet.
    # In single-month forecast mode keep the target month's rows even without obs,
    # but only at locations that are genuine SILO points (have obs at some time).
    # This has to happen before the index is rounded, which merges locations.
    has_obs = dataset["pr"].notnull()
    if is_forecast_month:
        t = pd.DatetimeIndex(dataset.index.get_level_values("time"))
        is_target = (t.year == target_year) & (t.month == target_month)
        loc_has_obs = has_obs.groupby(level=["latitude", "longitude"]).transform("any")
        keep = has_obs | (loc_has_obs & is_target)
    else:
        keep = has_obs
    dataset = dataset[keep]

    # Rounded Index to integer (upto 0.5 degree)

    if silo_radius > 0:
        rounded_index = pd.MultiIndex.from_tuples(
            [(round(t[0]), round(t[1]), t[2]) for t in dataset.index],
            names=dataset.index.names
        )
        dataset.index = rounded_index
    dataset = dataset.reset_index().drop(columns=["time"])

    if args.season == "wet":
        dataset = dataset[dataset["month"].isin([10, 11, 12, 1, 2, 3])]
    else:
        dataset = dataset[~dataset["month"].isin([10, 11, 12, 1, 2, 3])]
        # TODO: Check
        # Treat trace precipitation as effectively zero (< 0.1 mm/day)
        # dataset["pr"] = np.where(dataset["pr"] < 0.1, 0.0, dataset["pr"])

    dataset["pr"] = forward_transform(dataset["pr"], args.feature)  # NaN stays NaN

    # dataset = pd.get_dummies(dataset, columns=["month"], dtype=int)
    dataset = dataset.drop(columns=["month", "latitude", "longitude"])

    print(dataset)

    return dataset

def inverse_transform(values, feature):
    """Undo the cbrt/log feature transform to get back raw rainfall."""
    if feature == "log":
        return np.exp(values) - 1
    elif feature == "cbrt":
        return np.power(values, 3)
    raise AssertionError()


def forward_transform(values, feature):
    """Apply the cbrt/log feature transform to raw rainfall."""
    if feature == "log":
        return np.log(values + 1)
    elif feature == "cbrt":
        return np.cbrt(values)
    raise AssertionError()