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
| 3 | `ensemble_xgboost_nearest.py` | As above but using every point within 0.25° of the grid point, with `--region` / `--lead_time` / `--feature` / `--season` options, plus `--forecast` (operational mode: train on all hindcast years, score against the live forecast period, single train/val split instead of leave-one-year-out) and `--no_bias_correction` (skip quantile-mapping both the GCM inputs and the XGBoost output, for the before/after bias-correction comparison — this is what drives `hindcast/bias_analysis/before` vs `.../after`). This one script now covers hindcast, forecast, and before/after runs; it is the only actively-maintained training script — see "Superseded scripts" below. Best-performing variant. |
| 4 | `plot_heatmap.py` | Seaborn grid heatmaps of MAE/RMSE for all four models, plus an XGBoost − ECMWF difference plot. One figure per region. |
| 4 | `bias_analysis.py` | Seaborn boxplots and histograms for mean bias comparison before and after quantile mapping bias correction across the 3 models, XGBoost bias also shown after bias correction. Reads the `before`/`after` CSVs written by the two `ensemble_xgboost_nearest.py` `--no_bias_correction` runs above. (There is an older, unrelated `bias_analysis/bias_analysis.py` at the repo root — see "Superseded scripts" below; this is the current one.) |
| 4 | `plot_heatmap_cartopy.py` | The same metrics drawn on a Cartopy map with the Queensland boundary and coastlines, all regions on one figure. |
| 4 | `plot_lead_time_results.py` | The same metrics per region or aggregated across regions showing mean metric vs lead time, one line per model.  |

The three step-3 scripts are alternatives, not a sequence. Each writes `output_region{R}_lead{L}_feature{F}_perc{P}.csv`, which the step-4 plotting scripts read.

## Superseded scripts

`ensemble_xgboost_nearest.py` at the repo root is the only training script that should be run.
Two older scripts still exist in the tree but are **not** part of the pipeline above, are not
tracked by the pipeline table, and still contain the lead-time feature-leak described below
(their `feature_dataset.select_lead_time(args.lead_time)` call is commented out, unlike the
current `ensemble_xgboost_nearest.py`):

- `ensemble_xgboost_nearest_forecast.py` — superseded by `ensemble_xgboost_nearest.py --forecast`.
- `bias_analysis/ensemble_xgboost_nearest.py` (and its adjacent `bias_analysis/feature_engineering.py`,
  `bias_analysis/bias_correction.py`) — superseded by `ensemble_xgboost_nearest.py --no_bias_correction`.

Do not run either for new results; they are left in place for history only pending cleanup.

## Lead-time feature leak — cross-checked and fixed (24 Sept 2026)

`FeatureEngineering.__init__` keeps every lead from 0 up to the requested lead cumulatively
(`self.ds.isel(lead=slice(0, lead_months))`), by design — this is a feature-selection step from
the original design. At `--lead_time L`, that leaves leads `0..L-1` all present in `self.ds`
unless something narrows it back down to just lead `L-1`.

That narrowing is `FeatureEngineering.select_lead_time(lead_time)`
(`self.ds.isel(lead=[lead_time - 1])`), called right after construction, before
`create_lead_time_features()`. It was commented out in earlier versions, so XGBoost was trained
on leads `0..L-1` together instead of only lead `L-1` for any `--lead_time` > 1 — i.e. it could
see the much-more-accurate short-lead ("nowcast") forecast as an extra input feature even when
nominally predicting further ahead. This was invisible at `--lead_time 1`, where
`isel(lead=slice(0,1))` and `isel(lead=[0])` are identical, which is why lead-1 results were
always fine.

**Now fixed**: `ensemble_xgboost_nearest.py:184` calls `feature_dataset.select_lead_time(args.lead_time)`
unconditionally, before the hindcast/forecast branch, so both modes get it. Checked two ways:

1. **Trace-through**: after `isel(lead=slice(0, L))` then `isel(lead=[L-1])`, `isel` is positional
   and the first slice starts at 0, so position and label coincide throughout — the result is
   exactly the single lead `L-1`, matching the `..._{lead_time-1}_pr` column names the evaluation
   code reads. Confirmed for `L` = 1 through 6.
2. **Before/after data**: re-running region 3 (cbrt, perc95) hindcast before vs. after the fix
   changes the lead-dependence story materially — XGBoost's MAE is no longer near-flat across
   leads 2-6, and its win-rate against the best individual GCM drops from the pre-fix ~93-99%
   (leads 2-6) to roughly 41-53% at every lead, much closer to parity. The pre-fix numbers were
   an artefact of the leak, not genuine skill. See the emailed `hindcast_analysis.zip`
   (23-24 Sept 2026) for the corrected figures; any report built from data older than that date
   should be treated as superseded.

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

# Operational Forecast Results (Oct 2026-Mar 2027)
 
Forecast rainfall GIFs
 
Predicted rainfall for the operational forecast period, after bias correction (after), Queensland region, wet season.
 
## XGBoost
 
![XGBoost forecast rainfall](gifs_forecast_values/after/forecast_values_XGBoost_cbrt_region3_2026-10_to_2027-03.gif)
 
## ECMWF
 
![ECMWF forecast rainfall](gifs_forecast_values/after/forecast_values_ecmwf_cbrt_region3_2026-10_to_2027-03.gif)
 
## UKMO
 
![UKMO forecast rainfall](gifs_forecast_values/after/forecast_values_ukmo_cbrt_region3_2026-10_to_2027-03.gif)
 
## ACCESS
 
![ACCESS forecast rainfall](gifs_forecast_values/after/forecast_values_access_cbrt_region3_2026-10_to_2027-03.gif)