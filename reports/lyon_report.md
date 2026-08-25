
# Airbnb Price Prediction Report — Lyon

*Generated on 2026-08-25 by the automated pipeline (`make all`). Source: [InsideAirbnb](https://insideairbnb.com/get-the-data/).*

## Executive summary

This report analyzes the **Lyon** Airbnb market using 5,427 cleaned listings and 26 listing features. A machine-learning model was trained to predict nightly prices and then evaluated on a **holdout set** — listings it had never seen before.

The best model was **XGBoost**. On the holdout set it achieved:

- **Average error (MAE): $30** — a typical prediction is off by about this much.
- **RMSE: $49** — like MAE, but weighs large errors more heavily.
- **MAPE: 27%** — the average error as a share of the actual price.
- **R²: 0.70** — the share of price variation the model explains (0–1).
- **61%** of predictions were within +/-25% of the true price.


## How to read this report

- **MAE (mean absolute error)** — the average difference between predicted and actual prices, in dollars. Lower is better; it's the easiest number to grasp ("off by about $X on average").
- **RMSE (root mean squared error)** — similar to MAE but penalizes large mistakes more. If RMSE is much higher than MAE, a few big errors are inflating it.
- **MAPE (mean absolute percentage error)** — the same idea as MAE, but expressed as a percentage of the actual price, so it can be compared across cities.
- **R²** — how much of the variation in prices the model explains (0 = no better than guessing the average; 1 = perfect).
- **Holdout set** — the 10% of listings the model never saw while training. All accuracy numbers come from this set, so they are an honest estimate of real-world performance.
- **Top features** — the listing attributes the model leans on most when setting a price.

## Data

- Source: [InsideAirbnb](https://insideairbnb.com/get-the-data/) (latest available snapshot)
- Listings after cleaning: **5,427** — median nightly price **$104**, mean **$128**
- Features used by the model: **26**
- Train / validation / holdout split: **3,798 / 977 / 652** listings (70/15/10, random seed 42)

## Market overview

### Price distribution
![price_dist](artifacts/figures/lyon/price_dist.png)
- Median nightly price: **$104**; the middle 50% of listings fall between **$74** and **$149**.
- The mean (**$128**) sits well above the median — a minority of expensive listings pull the average up, so the **median is the better guide** to a typical price.

### Price by room type
![price_by_room_type](artifacts/figures/lyon/price_by_room_type.png)

| Room type | Listings | Median price | Mean price |
|---|---|---|---|
| Hotel room | 17 | $201 | $279 |
| Entire home/apt | 4,498 | $113 | $138 |
| Private room | 894 | $62 | $77 |
| Shared room | 18 | $31 | $35 |


### Price by number of bedrooms
![price_by_bedrooms](artifacts/figures/lyon/price_by_bedrooms.png)

### Feature correlations
![corr_heatmap](artifacts/figures/lyon/corr_heatmap.png)

## Model accuracy on the holdout set

Metrics for the chosen model (**XGBoost**) on 652 unseen listings:

| Metric | Value | What it means |
|---|---|---|
| MAE | $30 | Average prediction is off by this much |
| RMSE | $49 | Errors, with big misses weighted more |
| MAPE | 27% | Average error as a % of the true price |
| R² | 0.70 | Share of price variation explained |

**Accuracy in practice:** 27% of predictions were within +/-10% of the true price, 61% within +/-25%, and 87% within +/-50%.

![pred_vs_actual](artifacts/lyon/pred_vs_actual.png)
*Points close to the dashed line are accurate predictions; points above it mean the model under-predicted.*

![residuals](artifacts/lyon/residuals.png)
*Residual = actual - predicted. A cloud centered on the zero line means errors are balanced rather than systematically too high or too low.*


## Models & results

| Model | Validation RMSE |
|---|---|
| Random Forest | $57 |
| XGBoost | $52 |
| TabPFN (capped sample, not directly comparable) | $49 |

The model with the lowest validation RMSE was then scored once on the holdout set — see the accuracy section above for its final numbers.

## Top features

The listing attributes that influence price the most (importance = how much the model relies on them):

| Feature | Importance | Meaning |
|---|---|---|
| bedrooms | 0.180 | the number of bedrooms |
| bathrooms | 0.090 | the number of bathrooms |
| accommodates | 0.086 | how many guests a listing sleeps |
| room_type_Private room | 0.071 | the room type (entire home vs private/shared room) (specifically Private room) |
| minimum_nights | 0.059 | the minimum nights required |
| property_type_Private room in loft | 0.057 | the property type (apartment, house, etc.) (specifically Private room in loft) |
| property_type_Private room in rental unit | 0.057 | the property type (apartment, house, etc.) (specifically Private room in rental unit) |
| room_type_Hotel room | 0.049 | the room type (entire home vs private/shared room) (specifically Hotel room) |
| property_type_Room in hotel | 0.031 | the property type (apartment, house, etc.) (specifically Room in hotel) |
| room_type_Entire home/apt | 0.026 | the room type (entire home vs private/shared room) (specifically Entire home/apt) |


![shap_summary](artifacts/lyon/shap_summary.png)
*SHAP summary: each dot is one listing; colour shows the feature's value (red = high, blue = low) and position shows whether it pushed the predicted price up (right) or down (left).*

## Prediction examples

A few real holdout listings with the model's prediction:

| Listing | Actual | Predicted | Error | Error % |
|---|---|---|---|---|
| accommodates=2, bedrooms=1, room_type=Entire home/apt | $50 | $50 | $0 | 0.3% |
| accommodates=6, bedrooms=2, room_type=Entire home/apt | $222 | $345 | $123 | 55.2% |
| accommodates=4, bedrooms=2, room_type=Entire home/apt | $136 | $210 | $74 | 54.0% |
| accommodates=4, bedrooms=1, room_type=Entire home/apt | $112 | $117 | $6 | 5.2% |
| accommodates=7, bedrooms=3, room_type=Entire home/apt | $170 | $155 | $14 | 8.5% |
| accommodates=4, bedrooms=1, room_type=Entire home/apt | $88 | $101 | $13 | 14.6% |
| accommodates=4, bedrooms=2, room_type=Entire home/apt | $326 | $161 | $165 | 50.6% |
| accommodates=2, bedrooms=1, room_type=Entire home/apt | $124 | $112 | $12 | 9.6% |


## Recommendations


- The best-performing model was **XGBoost** (holdout RMSE $49, MAE $30, R² 0.70).

- On validation, **XGBoost** beat Random Forest by 8% on RMSE ($52 vs $57). A simple ensemble or blending the two may squeeze out further gains.

- Pricing is most strongly driven by **bedrooms, bathrooms, accommodates** — i.e. bedrooms (the number of bedrooms); bathrooms (the number of bathrooms); accommodates (how many guests a listing sleeps). Hosts can use these levers when setting nightly rates.

- On average, predictions are off by about **27%** of the actual price (median error 20%), and **61%** of predictions landed within +/-25% of the true price.

- Accuracy caveat: the holdout RMSE of $49 is **47% of the median listing price ($104)**, so predictions for typical listings are expected to be off by roughly +/-$49.

- Next steps: add richer features (neighbourhood_cleansed, amenities, review scores breakdowns), train on a log-transformed target to tame high-end outliers, run a deeper hyperparameter search, and extend the pipeline to more cities.


## Limitations

- The model only "sees" the features listed in the config — it cannot capture neighbourhood charm, amenities, photos, or seasonality.
- Extreme high-end prices are clipped at the 99th percentile during cleaning, so predictions at the very top of the market are less meaningful.
- The accuracy numbers are averages: individual predictions can be much better or much worse.
- This report is generated automatically from the pipeline artifacts and reflects the data as of 2026-08-25.

---
*Automated pipeline: fetch → preprocess → EDA → train → evaluate → report. Code in `src/`, config in `configs/base.yaml`.*