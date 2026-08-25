
# Airbnb Price Prediction Report - Lyon

*Generated on 2026-08-25 by the automated pipeline (`make all`).*

## Executive summary

The pipeline trained and compared Random Forest, XGBoost and TabPFN
on Airbnb listing data for **Lyon** (3798 listings).
The **xgboost** model was selected for final evaluation on a held-out test set and achieved:
- **RMSE: $49**
- **MAE: $30**
- **R²: 0.70**


## Data
- Source: InsideAirbnb (latest available snapshot). See: https://insideairbnb.com/get-the-data/
- City: Lyon · 3798 listings after cleaning

## EDA
![price_dist](artifacts/figures/lyon/price_dist.png)

- Median price: **$105** (mean $129)

![price_by_room_type](artifacts/figures/lyon/price_by_room_type.png)
![corr_heatmap](artifacts/figures/lyon/corr_heatmap.png)

## Models & Results

| Model | Validation RMSE |
|---|---|
| Random Forest | $57 |
| XGBoost | $52 |
| TabPFN (capped sample, not directly comparable) | $49 |

Holdout results (best model): see `artifacts/lyon/holdout_results.csv`.

## Top features

| Feature | Importance |
|---|---|
| bedrooms | 0.180 |
| bathrooms | 0.090 |
| accommodates | 0.086 |
| room_type_Private room | 0.071 |
| minimum_nights | 0.059 |
| property_type_Private room in loft | 0.057 |
| property_type_Private room in rental unit | 0.057 |
| room_type_Hotel room | 0.049 |
| property_type_Room in hotel | 0.031 |
| room_type_Entire home/apt | 0.026 |



## Recommendations

- The best-performing model was **xgboost** (holdout RMSE $49, MAE $30, R² 0.70).

- On validation, **XGBoost** beat Random Forest by 8% on RMSE ($52 vs $57). A simple ensemble or blending the two may squeeze out further gains.

- Pricing is most strongly driven by **bedrooms, bathrooms, accommodates** — i.e. bedrooms (the number of bedrooms); bathrooms (a key listing attribute); accommodates (how many guests a listing sleeps). Hosts can use these levers when setting nightly rates.

- Accuracy caveat: the holdout RMSE of $49 is **46% of the median listing price ($105)**, so predictions for typical listings are expected to be off by roughly ±$49.

- Next steps: add richer features (neighbourhood_cleansed, amenities, review scores breakdowns), train on a log-transformed target to tame high-end outliers, run a deeper hyperparameter search, and extend the pipeline to more cities.


---
*Automated pipeline: fetch → preprocess → EDA → train → evaluate → report. Code in `src/`, config in `configs/base.yaml`.*