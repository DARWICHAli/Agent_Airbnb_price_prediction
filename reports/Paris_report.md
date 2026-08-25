
# Airbnb Price Prediction Report — Paris

*Generated on 2026-08-25 by the automated pipeline (`make all`). Source: [InsideAirbnb](https://insideairbnb.com/get-the-data/).*

## Executive summary

This report analyzes the **Paris** Airbnb market using 48,402 cleaned listings and 26 listing features. A machine-learning model was trained to predict nightly prices and then evaluated on a **holdout set** — listings it had never seen before.

The best model was **XGBoost**. On the holdout set it achieved:

- **Average error (MAE): $84** — a typical prediction is off by about this much.
- **RMSE: $153** — like MAE, but weighs large errors more heavily.
- **MAPE: 33%** — the average error as a share of the actual price.
- **R²: 0.69** — the share of price variation the model explains (0–1).
- **55%** of predictions were within +/-25% of the true price.


## How to read this report

- **MAE (mean absolute error)** — the average difference between predicted and actual prices, in dollars. Lower is better; it's the easiest number to grasp ("off by about $X on average").
- **RMSE (root mean squared error)** — similar to MAE but penalizes large mistakes more. If RMSE is much higher than MAE, a few big errors are inflating it.
- **MAPE (mean absolute percentage error)** — the same idea as MAE, but expressed as a percentage of the actual price, so it can be compared across cities.
- **R²** — how much of the variation in prices the model explains (0 = no better than guessing the average; 1 = perfect).
- **Holdout set** — the 10% of listings the model never saw while training. All accuracy numbers come from this set, so they are an honest estimate of real-world performance.
- **Top features** — the listing attributes the model leans on most when setting a price.

## Data

- Source: [InsideAirbnb](https://insideairbnb.com/get-the-data/) (latest available snapshot)
- Listings after cleaning: **48,402** — median nightly price **$206**, mean **$293**
- Features used by the model: **26**
- Train / validation / holdout split: **33,881 / 8,712 / 5,809** listings (70/15/10, random seed 42)

## Market overview

### Price distribution
![price_dist](artifacts/figures/Paris/price_dist.png)
- Median nightly price: **$206**; the middle 50% of listings fall between **$131** and **$342**.
- The mean (**$293**) sits well above the median — a minority of expensive listings pull the average up, so the **median is the better guide** to a typical price.

### Price by room type
![price_by_room_type](artifacts/figures/Paris/price_by_room_type.png)

| Room type | Listings | Median price | Mean price |
|---|---|---|---|
| Hotel room | 310 | $298 | $386 |
| Entire home/apt | 43,131 | $213 | $301 |
| Private room | 4,848 | $137 | $226 |
| Shared room | 113 | $66 | $88 |


### Price by number of bedrooms
![price_by_bedrooms](artifacts/figures/Paris/price_by_bedrooms.png)

### Feature correlations
![corr_heatmap](artifacts/figures/Paris/corr_heatmap.png)

## Model accuracy on the holdout set

Metrics for the chosen model (**XGBoost**) on 5,809 unseen listings:

| Metric | Value | What it means |
|---|---|---|
| MAE | $84 | Average prediction is off by this much |
| RMSE | $153 | Errors, with big misses weighted more |
| MAPE | 33% | Average error as a % of the true price |
| R² | 0.69 | Share of price variation explained |

**Accuracy in practice:** 23% of predictions were within +/-10% of the true price, 55% within +/-25%, and 82% within +/-50%.

![pred_vs_actual](artifacts/Paris/pred_vs_actual.png)
*Points close to the dashed line are accurate predictions; points above it mean the model under-predicted.*

![residuals](artifacts/Paris/residuals.png)
*Residual = actual - predicted. A cloud centered on the zero line means errors are balanced rather than systematically too high or too low.*


## Models & results

| Model | Validation RMSE |
|---|---|
| Random Forest | $173 |
| XGBoost | $158 |
| TabPFN (capped sample, not directly comparable) | $175 |

The model with the lowest validation RMSE was then scored once on the holdout set — see the accuracy section above for its final numbers.

## Top features

The listing attributes that influence price the most (importance = how much the model relies on them):

| Feature | Importance | Meaning |
|---|---|---|
| bathrooms | 0.193 | the number of bathrooms |
| accommodates | 0.084 | how many guests a listing sleeps |
| bedrooms | 0.078 | the number of bedrooms |
| availability_90 | 0.061 | availability over the next 90 days |
| minimum_nights | 0.049 | the minimum nights required |
| property_type_Room in boutique hotel | 0.044 | the property type (apartment, house, etc.) (specifically Room in boutique hotel) |
| room_type_Entire home/apt | 0.043 | the room type (entire home vs private/shared room) (specifically Entire home/apt) |
| property_type_Private room in rental unit | 0.028 | the property type (apartment, house, etc.) (specifically Private room in rental unit) |
| property_type_Room in hotel | 0.026 | the property type (apartment, house, etc.) (specifically Room in hotel) |
| room_type_Private room | 0.026 | the room type (entire home vs private/shared room) (specifically Private room) |


![shap_summary](artifacts/Paris/shap_summary.png)
*SHAP summary: each dot is one listing; colour shows the feature's value (red = high, blue = low) and position shows whether it pushed the predicted price up (right) or down (left).*

## Prediction examples

A few real holdout listings with the model's prediction:

| Listing | Actual | Predicted | Error | Error % |
|---|---|---|---|---|
| accommodates=2, bedrooms=1, room_type=Entire home/apt | $215 | $240 | $25 | 11.8% |
| accommodates=1, bedrooms=1, room_type=Entire home/apt | $47 | $32 | $15 | 32.0% |
| accommodates=11, bedrooms=2, room_type=Entire home/apt | $944 | $1,094 | $150 | 15.9% |
| accommodates=2, bedrooms=1, room_type=Entire home/apt | $110 | $116 | $6 | 5.0% |
| accommodates=2, bedrooms=1, room_type=Entire home/apt | $51 | $79 | $28 | 55.7% |
| accommodates=2, bedrooms=1, room_type=Entire home/apt | $183 | $150 | $32 | 17.6% |
| accommodates=2, bedrooms=1, room_type=Entire home/apt | $291 | $283 | $8 | 2.8% |
| accommodates=3, bedrooms=1, room_type=Entire home/apt | $500 | $205 | $295 | 58.9% |


## Recommendations


- The best-performing model was **XGBoost** (holdout RMSE $153, MAE $84, R² 0.69).

- On validation, **XGBoost** beat Random Forest by 9% on RMSE ($158 vs $173). A simple ensemble or blending the two may squeeze out further gains.

- Pricing is most strongly driven by **bathrooms, accommodates, bedrooms** — i.e. bathrooms (the number of bathrooms); accommodates (how many guests a listing sleeps); bedrooms (the number of bedrooms). Hosts can use these levers when setting nightly rates.

- On average, predictions are off by about **33%** of the actual price (median error 22%), and **55%** of predictions landed within +/-25% of the true price.

- Accuracy caveat: the holdout RMSE of $153 is **75% of the median listing price ($206)**, so predictions for typical listings are expected to be off by roughly +/-$153.

- Next steps: add richer features (neighbourhood_cleansed, amenities, review scores breakdowns), train on a log-transformed target to tame high-end outliers, run a deeper hyperparameter search, and extend the pipeline to more cities.


## Limitations

- The model only "sees" the features listed in the config — it cannot capture neighbourhood charm, amenities, photos, or seasonality.
- Extreme high-end prices are clipped at the 99th percentile during cleaning, so predictions at the very top of the market are less meaningful.
- The accuracy numbers are averages: individual predictions can be much better or much worse.
- This report is generated automatically from the pipeline artifacts and reflects the data as of 2026-08-25.

---
*Automated pipeline: fetch → preprocess → EDA → train → evaluate → report. Code in `src/`, config in `configs/base.yaml`.*