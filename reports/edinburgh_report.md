
# Airbnb Price Prediction Report - Edinburgh

*Generated on 2026-08-25 by the automated pipeline (`make all`).*

## Executive summary

The pipeline trained and compared Random Forest, XGBoost and TabPFN
on Airbnb listing data for **Edinburgh** (3447 listings).
The **xgboost** model was selected for final evaluation on a held-out test set and achieved:
- **RMSE: $162**
- **MAE: $68**
- **R²: 0.64**


## Data
- Source: InsideAirbnb (latest available snapshot). See: https://insideairbnb.com/get-the-data/
- City: Edinburgh · 3447 listings after cleaning

## EDA
![price_dist](artifacts/figures/edinburgh/price_dist.png)

- Median price: **$160** (mean $215)

![price_by_room_type](artifacts/figures/edinburgh/price_by_room_type.png)
![corr_heatmap](artifacts/figures/edinburgh/corr_heatmap.png)

## Models & Results

| Model | Validation RMSE |
|---|---|
| Random Forest | $137 |
| XGBoost | $123 |
| TabPFN (capped sample, not directly comparable) | $122 |

Holdout results (best model): see `artifacts/edinburgh/holdout_results.csv`.

## Top features

| Feature | Importance |
|---|---|
| property_type_Private room in townhouse | 0.083 |
| host_neighbourhood_New Town | 0.078 |
| host_neighbourhood_Cannonmills | 0.071 |
| bedrooms | 0.063 |
| host_neighbourhood_Fasach | 0.057 |
| host_neighbourhood_Laulerie | 0.042 |
| accommodates | 0.037 |
| availability_90 | 0.034 |
| bathrooms | 0.034 |
| host_listings_count | 0.028 |



## Recommendations

- The best-performing model was **xgboost** (holdout RMSE $162, MAE $68, R² 0.64).

- On validation, **XGBoost** beat Random Forest by 10% on RMSE ($123 vs $137). A simple ensemble or blending the two may squeeze out further gains.

- Pricing is most strongly driven by **property_type_Private room in townhouse, host_neighbourhood_New Town, host_neighbourhood_Cannonmills** — i.e. property_type_Private room in townhouse (the property type (apartment, house, etc.) (specifically Private room in townhouse)); host_neighbourhood_New Town (the host's neighbourhood (specifically New Town)); host_neighbourhood_Cannonmills (the host's neighbourhood (specifically Cannonmills)). Hosts can use these levers when setting nightly rates.

- Accuracy caveat: the holdout RMSE of $162 is **101% of the median listing price ($160)**, so predictions for typical listings are expected to be off by roughly ±$162.

- Next steps: add richer features (neighbourhood_cleansed, amenities, review scores breakdowns), train on a log-transformed target to tame high-end outliers, run a deeper hyperparameter search, and extend the pipeline to more cities.


---
*Automated pipeline: fetch → preprocess → EDA → train → evaluate → report. Code in `src/`, config in `configs/base.yaml`.*