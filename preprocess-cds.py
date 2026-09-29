"""
Preprocess CDS seasonal forecast datasets for climpred.

Converted from preprocess-cds.ipynb. Loops over multiple datasets
(ecmwf_51, ukmo_604, bom_2) for a given region ("range_id" / area index),
applying the same processing pipeline to each:

    1. Open the raw NetCDF file (mask_and_scale=False, then decode_cf)
    2. Convert tprate units to mm/day
    3. Rename coords/vars to climpred conventions (init, lead, member, pr)
    4. Convert from init+lead indexing to valid_time+lead indexing
    5. Rename valid_time -> time, drop the now-redundant init coord
    6. Take the ensemble median over 'member'
    7. Write out the processed NetCDF file
"""

import climpred as cp
import xarray as xr
from xarray import DataArray

range_id = 3
dataset_names = ["ecmwf_51", "ukmo_604", "bom_2"]
is_forecast = True

INPUT_DIR = "/scratch/tm70/ag9761/iceds_datasets"
OUTPUT_DIR = "/scratch/tm70/ag9761/iceds_datasets"

# Model tag -> processed filename prefix, used when merging.
MODEL_FILE_PREFIXES = {
    "ecmwf": "ecmwf_51",
    "ukmo": "ukmo_604", 
    "access": "bom_2",
}


def convert_units(da: DataArray, new_unit: str) -> DataArray:
    da = da.metpy.quantify()
    da = da.metpy.convert_units(new_unit)
    da = da.metpy.dequantify()
    return da


def process_dataset(dataset_name: str, range_id: int, is_forecast: bool = False) -> None:
    print(f"Processing {dataset_name} (region {range_id}, forecast={is_forecast})...")

    suffix = "_forecast" if is_forecast else ""

    # Mask and scale is used for scaling factor
    input_path = f"{INPUT_DIR}/{dataset_name}_{range_id}_new{suffix}.nc"
    dataset = xr.open_dataset(input_path, mask_and_scale=False, chunks="auto")
    dataset = xr.decode_cf(dataset)

    # Convert precipitation rate to mm/day
    dataset["tprate"] = convert_units(dataset["tprate"], "mm/day")

    # Make dataset ready for climpred
    renamed_vars = ["indexing_time", "forecast_reference_time"]
    for rv in renamed_vars:
        if rv in dataset.coords:
            dataset = dataset.rename({rv: "init"})
    dataset = dataset.rename(
        {"forecastMonth": "lead", "number": "member", "tprate": "pr"}
    )
    dataset["lead"].attrs["units"] = "months"
    dataset = dataset.assign_coords(lead=dataset["lead"] - 1)

    # Convert from init + lead config to valid_time + lead config
    model = cp.HindcastEnsemble(dataset).get_initialized()
    dataset = cp.utils.convert_init_lead_to_valid_time_lead(model)

    dataset = dataset.rename({"valid_time": "time"})
    dataset = dataset.drop_vars(["init"])

    # Prediction on ensemble median
    dataset = dataset.median(dim="member")

    output_path = f"{OUTPUT_DIR}/{dataset_name}_{range_id}_new_processed{suffix}.nc"
    dataset.to_netcdf(output_path)
    print(f"  -> wrote {output_path}")

def merge_datasets(range_id: int, is_forecast: bool = False) -> None:
    """
    Merge the processed ecmwf, ukmo, and bom (access) datasets for a given
    region into a single dataset stacked along a new 'model' dimension, and
    write it out.
 
    Converted from merge_datasets.ipynb. Note: the SILO observational
    dataset is loaded (as in the notebook) but, matching the notebook's
    final concat cell, is not included in the merged output below.
    """
    print(f"Merging processed datasets for region {range_id} (forecast={is_forecast})...")
 
    suffix = "_forecast" if is_forecast else ""
 
    ecmwf = xr.open_dataset(
        f"{INPUT_DIR}/{MODEL_FILE_PREFIXES['ecmwf']}_{range_id}_new_processed{suffix}.nc",
        chunks="auto",
    )
    ecmwf = ecmwf.assign_coords(model="ecmwf")
 
    ukmo = xr.open_dataset(
        f"{INPUT_DIR}/{MODEL_FILE_PREFIXES['ukmo']}_{range_id}_new_processed{suffix}.nc",
        chunks="auto",
    )
    ukmo = ukmo.assign_coords(model="ukmo")
 
    access = xr.open_dataset(
        f"{INPUT_DIR}/{MODEL_FILE_PREFIXES['access']}_{range_id}_new_processed{suffix}.nc",
        chunks="auto",
    )
    access = access.assign_coords(model="access")
 
    # Loaded for parity with the notebook; not included in the merged
    # output (the notebook's concat only combines ecmwf/ukmo/access).
    # SILO is observational, not a forecast product, so its path is never
    # suffixed with "_forecast" regardless of the is_forecast flag.
    silo_path = f"{INPUT_DIR}/silo_monthly_rain_iceds_{range_id}_processed.nc"
    try:
        silo = xr.open_dataset(silo_path, chunks="auto")
        silo = silo.assign_coords(model="silo")
    except FileNotFoundError:
        print(f"  (skipping SILO: {silo_path} not found)")
 
    preprocessed_dataset = xr.concat([ecmwf, ukmo, access], dim="model", coords="minimal")
 
    output_path = f"{OUTPUT_DIR}/preprocessed_dataset_{range_id}_new{suffix}.nc"

    if is_forecast:
        preprocessed_dataset = preprocessed_dataset.isel(time=slice(None, 12))

    preprocessed_dataset.to_netcdf(output_path)
    print(f"  -> wrote {output_path}")
 
 
def main():
    for dataset_name in dataset_names:
        process_dataset(dataset_name, range_id, is_forecast)
 
    merge_datasets(range_id, is_forecast)

if __name__ == "__main__":
    main()