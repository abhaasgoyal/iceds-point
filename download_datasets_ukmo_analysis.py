import os

import cdsapi

is_forecast = False  # hindcast

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
    {"area": [-10, 138, -29, 153]}
]

areas = [
    {"area": [area["area"][0] + 3, area["area"][1] - 3, area["area"][2] - 3, area["area"][3] + 3]}
    for area in areas
]

print(areas)

# ---------------------------------------------------------------------------
# UKMO hindcasts, Jan 1993 - Dec 2016 (all start months), for each system.
#
# Unlike the forecast script, the systems are NOT combined: each system has
# its own hindcast set, used to calibrate/bias-correct forecasts from that
# same system. So each system gets its own output file per area.
# ---------------------------------------------------------------------------

HINDCAST_YEARS = [str(y) for y in range(1993, 2017)]  # 1993..2016 inclusive
ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]

models = [
    {
        "originating_centre": "ukmo",
        "systems": ["604", "605", "610"],
    },
]

OUT_DIR = "/scratch/tm70/ag9761"

extension = "nc" if base_request["data_format"] == "netcdf" else "grib"


def download_datasets():
    client = cdsapi.Client()
    for model in models:
        centre = model["originating_centre"]
        for system in model["systems"]:
            for i, area in enumerate(areas):
                target = (
                    f"{OUT_DIR}/{centre}_{system}_{i}_new"
                    f"{'_forecast' if is_forecast else '_hindcast'}_analysis.{extension}"
                )
                if os.path.exists(target):
                    print(f"Skipping {target} (already exists)")
                    continue

                req = base_request | area | {
                    "originating_centre": centre,
                    "system": system,
                    "year": HINDCAST_YEARS,
                    "month": ALL_MONTHS,
                }
                print(f"Requesting {centre} system {system}, area {i}")
                client.retrieve(dataset, req).download(target)
                print(f"Wrote {target}")


if __name__ == "__main__":
    download_datasets()