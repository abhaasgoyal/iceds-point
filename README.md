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