
# Airbnb Price Prediction Report - New York City

*Generated on 2026-08-25 by the automated pipeline (`make all`).*

## Executive summary

The pipeline trained and compared Random Forest, XGBoost and TabPFN
on Airbnb listing data for **New York City** (14929 listings).
The **xgboost** model was selected for final evaluation on a held-out test set and achieved:
- **RMSE: $572**
- **MAE: $116**
- **R²: 0.92**


## Data
- Source: InsideAirbnb (latest available snapshot). See: https://insideairbnb.com/get-the-data/
- City: New York City · 14929 listings after cleaning

## EDA
![price_dist](artifacts/figures/new-york-city/price_dist.png)

- Median price: **$154** (mean $420)

![price_by_room_type](artifacts/figures/new-york-city/price_by_room_type.png)
![corr_heatmap](artifacts/figures/new-york-city/corr_heatmap.png)

## Models & Results

| Model | Validation RMSE |
|---|---|
| Random Forest | $713 |
| XGBoost | $667 |
| TabPFN (capped sample, not directly comparable) | $956 |

Holdout results (best model): see `artifacts/new-york-city/holdout_results.csv`.

## Top features

| Feature | Importance |
|---|---|
| room_type_Hotel room | 0.400 |
| host_response_time_within an hour | 0.134 |
| host_neighbourhood_Gateway District | 0.084 |
| host_neighbourhood_Central Business District | 0.049 |
| host_months_since | 0.027 |
| host_neighbourhood_Airport North | 0.025 |
| property_type_Room in hotel | 0.022 |
| beds | 0.021 |
| host_neighbourhood_Clearwater Beach | 0.019 |
| minimum_nights | 0.018 |



## Recommendations

- The best-performing model was **xgboost** (holdout RMSE $572, MAE $116, R² 0.92).

- On validation, **XGBoost** beat Random Forest by 6% on RMSE ($667 vs $713). A simple ensemble or blending the two may squeeze out further gains.

- Pricing is most strongly driven by **room_type_Hotel room, host_response_time_within an hour, host_neighbourhood_Gateway District** — i.e. room_type_Hotel room (the room type (entire home vs private/shared room) (specifically Hotel room)); host_response_time_within an hour (how quickly the host responds (specifically within an hour)); host_neighbourhood_Gateway District (the host's neighbourhood (specifically Gateway District)). Hosts can use these levers when setting nightly rates.

- Accuracy caveat: the holdout RMSE of $572 is **371% of the median listing price ($154)**, so predictions for typical listings are expected to be off by roughly ±$572.

- Next steps: add richer features (neighbourhood_cleansed, amenities, review scores breakdowns), train on a log-transformed target to tame high-end outliers, run a deeper hyperparameter search, and extend the pipeline to more cities.


---
*Automated pipeline: fetch → preprocess → EDA → train → evaluate → report. Code in `src/`, config in `configs/base.yaml`.*