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
