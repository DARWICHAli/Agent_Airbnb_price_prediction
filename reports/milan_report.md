
# Airbnb Price Prediction Report - Milan

*Generated on 2026-08-25 by the automated pipeline (`make all`).*

## Executive summary

The pipeline trained and compared Random Forest, XGBoost and TabPFN
on Airbnb listing data for **Milan** (14317 listings).
The **xgboost** model was selected for final evaluation on a held-out test set and achieved:
- **RMSE: $107**
- **MAE: $54**
- **R²: 0.50**


## Data
- Source: InsideAirbnb (latest available snapshot). See: https://insideairbnb.com/get-the-data/
- City: Milan · 14317 listings after cleaning

## EDA
![price_dist](artifacts/figures/milan/price_dist.png)

- Median price: **$108** (mean $152)

![price_by_room_type](artifacts/figures/milan/price_by_room_type.png)
![corr_heatmap](artifacts/figures/milan/corr_heatmap.png)

## Models & Results

| Model | Validation RMSE |
|---|---|
| Random Forest | $121 |
| XGBoost | $112 |
| TabPFN (capped sample, not directly comparable) | $108 |

Holdout results (best model): see `artifacts/milan/holdout_results.csv`.

## Top features

| Feature | Importance |
|---|---|
| room_type_Entire home/apt | 0.053 |
| room_type_Private room | 0.052 |
| bedrooms | 0.045 |
| bathrooms | 0.044 |
| instant_bookable | 0.030 |
| host_response_time_nan | 0.030 |
| accommodates | 0.026 |
| host_neighbourhood_South Kensington | 0.026 |
| host_listings_count | 0.025 |
| host_total_listings_count | 0.025 |



## Recommendations

- The best-performing model was **xgboost** (holdout RMSE $107, MAE $54, R² 0.50).

- On validation, **XGBoost** beat Random Forest by 7% on RMSE ($112 vs $121). A simple ensemble or blending the two may squeeze out further gains.

- Pricing is most strongly driven by **room_type_Entire home/apt, room_type_Private room, bedrooms** — i.e. room_type_Entire home/apt (the room type (entire home vs private/shared room) (specifically Entire home/apt)); room_type_Private room (the room type (entire home vs private/shared room) (specifically Private room)); bedrooms (the number of bedrooms). Hosts can use these levers when setting nightly rates.

- Accuracy caveat: the holdout RMSE of $107 is **99% of the median listing price ($108)**, so predictions for typical listings are expected to be off by roughly ±$107.

- Next steps: add richer features (neighbourhood_cleansed, amenities, review scores breakdowns), train on a log-transformed target to tame high-end outliers, run a deeper hyperparameter search, and extend the pipeline to more cities.


---
*Automated pipeline: fetch → preprocess → EDA → train → evaluate → report. Code in `src/`, config in `configs/base.yaml`.*