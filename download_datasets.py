import cdsapi

dataset = "seasonal-monthly-single-levels"
request = {
    "variable": ["total_precipitation"],
    "year": [y for y in range(1993, 2017)],
    "month": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
    "leadtime_month": ["1", "2", "3", "4", "5", "6"],
    "data_format": "netcdf",
    "grid": [0.05, 0.05],
    "product_type": ["monthly_mean"],
}

areas = [
    { "area" : [ -22, 142, -25, 145 ] },
    { "area" : [ -26, 150, -29, 153 ] },
    { "area" : [ -17, 143, -20, 146 ] }
]

areas = [
   {"area": [area["area"][0] + 3, area["area"][1] - 3, area["area"][2] - 3, area["area"][3] + 3] }  for area in areas
]

print(areas)

models = [
    {
        "originating_centre": "ukmo",
        "system": "604",
    },
    {
        "originating_centre": "ecmwf",
        "system": "51",
    },
    {
        "originating_centre": "bom",
        "system": "2",
    },
]

extension = "nc" if request["data_format"] == "netcdf" else "grib"

def download_datasets():
    client = cdsapi.Client()
    for model in models:
        for i, area in enumerate(areas):
            client.retrieve(dataset, request | model | area).download(
                target=f"{model['originating_centre']}_{model['system']}_{i}_new.{extension}"
            )


download_datasets()
