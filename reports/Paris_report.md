
# Airbnb Price Prediction Report - Paris

*Generated on 2026-08-25 by the automated pipeline (`make all`).*

## Executive summary

The pipeline trained and compared Random Forest, XGBoost and TabPFN
on Airbnb listing data for **Paris** (33881 listings).
The **xgboost** model was selected for final evaluation on a held-out test set and achieved:
- **RMSE: $153**
- **MAE: $84**
- **R²: 0.69**


## Data
- Source: InsideAirbnb (latest available snapshot). See: https://insideairbnb.com/get-the-data/
- City: Paris · 33881 listings after cleaning

## EDA
![price_dist](artifacts/figures/Paris/price_dist.png)

- Median price: **$206** (mean $294)

![price_by_room_type](artifacts/figures/Paris/price_by_room_type.png)
![corr_heatmap](artifacts/figures/Paris/corr_heatmap.png)

## Models & Results

| Model | Validation RMSE |
|---|---|
| Random Forest | $173 |
| XGBoost | $158 |
| TabPFN (capped sample, not directly comparable) | $175 |

Holdout results (best model): see `artifacts/Paris/holdout_results.csv`.

## Top features

| Feature | Importance |
|---|---|
| bathrooms | 0.193 |
| accommodates | 0.084 |
| bedrooms | 0.078 |
| availability_90 | 0.061 |
| minimum_nights | 0.049 |
| property_type_Room in boutique hotel | 0.044 |
| room_type_Entire home/apt | 0.043 |
| property_type_Private room in rental unit | 0.028 |
| property_type_Room in hotel | 0.026 |
| room_type_Private room | 0.026 |



## Recommendations

- The best-performing model was **xgboost** (holdout RMSE $153, MAE $84, R² 0.69).

- On validation, **XGBoost** beat Random Forest by 9% on RMSE ($158 vs $173). A simple ensemble or blending the two may squeeze out further gains.

- Pricing is most strongly driven by **bathrooms, accommodates, bedrooms** — i.e. bathrooms (a key listing attribute); accommodates (how many guests a listing sleeps); bedrooms (the number of bedrooms). Hosts can use these levers when setting nightly rates.

- Accuracy caveat: the holdout RMSE of $153 is **75% of the median listing price ($206)**, so predictions for typical listings are expected to be off by roughly ±$153.

- Next steps: add richer features (neighbourhood_cleansed, amenities, review scores breakdowns), train on a log-transformed target to tame high-end outliers, run a deeper hyperparameter search, and extend the pipeline to more cities.


---
*Automated pipeline: fetch → preprocess → EDA → train → evaluate → report. Code in `src/`, config in `configs/base.yaml`.*