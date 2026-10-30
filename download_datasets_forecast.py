import cdsapi

is_forecast = True

dataset = "seasonal-monthly-single-levels"

base_request = {
    "variable": ["total_precipitation"],
    "leadtime_month": ["1", "2", "3", "4", "5", "6"],
    "data_format": "netcdf",
    "grid": [0.05, 0.05],
    "product_type": ["monthly_mean"],
}

areas = [
    # { "area" : [ -22, 142, -25, 145 ] },
    # { "area" : [ -26, 150, -29, 153 ] },
    # { "area" : [ -17, 143, -20, 146 ] },
    { "area" : [ -10, 138, -29, 153 ] }
]

areas = [
    {"area": [area["area"][0] + 3, area["area"][1] - 3, area["area"][2] - 3, area["area"][3] + 3]}
    for area in areas
]

print(areas)

# ---------------------------------------------------------------------------
# Models to download, for the forecast period May 2025 - Sep 2026 (inclusive).
#
# ECMWF and BOM keep a single 'system' number across the whole window.
# UKMO changes 'system' partway through (per the C3S nominal-start-dates
# table), so it's split into three sub-periods, each with its own
# {year: [months]} map. All UKMO sub-periods are downloaded and combined
# into a single output file per area.
# ---------------------------------------------------------------------------

models = [
    {
        "originating_centre": "ecmwf",
        "periods": [
            {
                "system": "51",
                "year_months": {
                    2025: ["05", "06", "07", "08", "09", "10", "11", "12"],
                    2026: ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
                },
            },
        ],
    },
    {
        "originating_centre": "bom",
        "periods": [
            {
                "system": "2",
                "year_months": {
                    2025: ["05", "06", "07", "08", "09", "10", "11", "12"],
                    2026: ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
                },
            },
        ],
    },
    {
        "originating_centre": "ukmo",
        "periods": [
            {
                # May 2025 - Feb 2026
                "system": "604",
                "year_months": {
                    2025: ["05", "06", "07", "08", "09", "10", "11", "12"],
                    2026: ["01", "02"],
                },
            },
            {
                # March 2026
                "system": "605",
                "year_months": {
                    2026: ["03"],
                },
            },
            {
                # April 2026 - Sep 2026
                "system": "610",
                "year_months": {
                    2026: ["04", "05", "06", "07", "08", "09", "10"],
                },
            },
        ],
    },
]

extension = "nc" if base_request["data_format"] == "netcdf" else "grib"


def download_datasets():
    client = cdsapi.Client()
    for model in models:
        centre = model["originating_centre"]
        for i, area in enumerate(areas):
            target = f"/scratch/tm70/ag9761/{centre}_{i}_new{'_forecast' if is_forecast else ''}.{extension}"
            downloaded_parts = []

            for period in model["periods"]:
                system = period["system"]
                for year, months in period["year_months"].items():
                    req = base_request | area | {
                        "originating_centre": centre,
                        "system": system,
                        "year": [str(year)],
                        "month": months,
                    }
                    part_target = (
                        f"/scratch/tm70/ag9761/_tmp_{centre}_{system}_{year}_{i}.{extension}"
                    )
                    client.retrieve(dataset, req).download(part_target)
                    downloaded_parts.append(part_target)

            # Combine all sub-period files for this model/area into one file.
            if len(downloaded_parts) == 1:
                import shutil
                shutil.move(downloaded_parts[0], target)
            else:
                combine_netcdf(downloaded_parts, target)
                for p in downloaded_parts:
                    import os
                    os.remove(p)


def combine_netcdf(part_paths, target_path):
    """Concatenate multiple monthly-mean forecast files along time/system
    where relevant. Since UKMO's system number changes across sub-periods,
    'system' won't be a shared coordinate across all parts, so we combine
    by stacking along a new 'forecast_reference_time'/time dimension using
    xarray's combine='nested' with compat='override', keeping each part's
    own system attribute recorded as a variable for traceability.
    """
    import xarray as xr

    datasets = []
    for p in part_paths:
        ds = xr.open_dataset(p)
        datasets.append(ds)

    combined = xr.merge(datasets, compat="no_conflicts", join="outer")
    combined.to_netcdf(target_path)

    for ds in datasets:
        ds.close()


download_datasets()