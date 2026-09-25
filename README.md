# Pipeline

Scripts in the order they are run. `feature_engineering.py` and `bias_correction.py` are imported by the training scripts rather than run directly.

| # | Script | Purpose |
|---|--------|---------|
| 1 | `download_datasets.py` | Downloads raw ECMWF/UKMO/BoM seasonal forecast hindcast NetCDFs from the CDS API, one file per model per region. |
| 1 | `download_datasets_forecast.py` | Downloads raw ECMWF/UKMO/BoM seasonal forecast NetCDFs from the CDS API for whole Queensland region. |
| 1 | `silo/silo_dataset.sh` | Download raw observation (Queensland SILO) precipitation data across years and collates them in a single file |
| 2 | `preprocess-cds.py` | Converts each raw model file to climpred's `valid_time`+`lead` layout (mm/day, ensemble median), then merges the three models into `preprocessed_dataset_{region}_new.nc`. |
| – | `feature_engineering.py` | Library. Builds one feature column per model/lead (`ecmwf_0_pr`, `ukmo_1_pr`, …) and assembles the training frame: merge with SILO, wet/dry season filter, cbrt or log transform of rainfall. |
| – | `bias_correction.py` | Library. Quantile-maps each GCM forecast column onto the observations, fitting only on non-extreme obs/simh pairs and optionally extrapolating the fit onto extremes. |
| 3 | `ensemble_xgboost.py` | Trains XGBoost on hexagonal grid points around each location. Lead 0 only. |
| 3 | `ensemble_xgboost_nearest.py` | As above but using every point within 0.25° of the grid point, with `--region` / `--lead_time` / `--feature` / `--season` options. Best-performing variant. |
| 4 | `plot_heatmap.py` | Seaborn grid heatmaps of MAE/RMSE for all four models, plus an XGBoost − ECMWF difference plot. One figure per region. |
| 4 | `bias_analysis.py` | Seaborn boxplots and histograms for mean bias comparison before and after quantile mapping bias correction across the 3 models, XGBoost bias also shown after bias correction |
| 4 | `plot_heatmap_cartopy.py` | The same metrics drawn on a Cartopy map with the Queensland boundary and coastlines, all regions on one figure. |
| 4 | `plot_lead_time_results.py` | The same metrics per region or aggregated across regions showing mean metric vs lead time, one line per model.  |

The three step-3 scripts are alternatives, not a sequence. Each writes `output_region{R}_lead{L}_feature{F}_perc{P}.csv`, which the step-4 plotting scripts read.

# Data Preprocessing

Original Datasets

| Model     | Resolution                        |
|-----------|-----------------------------------|
| ECMWF     | TCO_{319}/L91                     |
| GloSea6   | N216/L85                          |
| ACCESS-S2 | N216 (~60km in the mid-latitudes) |

We decided to match all data to 0.05° resolution grid cell (same as original SILO observation). For that purpose, we use `grid` key from CDSAPI specifying resolution. The resulting dataset is interpolated by bilinear interpolation. [1]

For example, let's assume that the original GCM dataset was in 0.6° resolution. If we were given a point of `[-23, 143]`, the interpolation would be done using the bounding box:

| Corner | Latitude | Longitude |
|--------|----------|-----------|
| SW | -23.4 | 142.8 |
| SE | -23.4 | 143.4 |
| NW | -22.8 | 142.8 |
| NE | -22.8 | 143.4 |

Now for getting additional data points to train on aside from the single location, we cherry pick points using methodologies as follows:

1. `ensemble_xgboost_nearest.py` uses all points which are less than or equal to 0.25° from the grid-point
2. `ensemble_xgboost.py` can use hexagonal grids at a specified distance (0.2° or 0.6°) or it's multiples. If the grid cell doesn't exactly match with the circular point, we take the nearest point here.

Current results (best-performing as of now) are generated from `ensemble_xgboost_nearest.py` 

[1] https://confluence.ecmwf.int/spaces/CKB/pages/56661594/Does+downloading+data+at+higher+resolution+improve+the+output


TODO: Look into the results for the following (quoting Barry):
```
The grid cells at -27,153 and -26, 153 have higher errors also (though not as high as the others near Cairns). These are north of Brisbane and fairly close to the coast. Leave these in for now, but these might also need to be flagged after we have done some exploration. It is worth noting that the two cells immediately west of these have also slightly elevated errors (though less than the coastal cells)
```