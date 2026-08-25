
# Airbnb Price Prediction Report — Milan

*Generated on 2026-08-25 by the automated pipeline (`make all`). Source: [InsideAirbnb](https://insideairbnb.com/get-the-data/).*

## Executive summary

This report analyzes the **Milan** Airbnb market using 20,453 cleaned listings and 26 listing features. A machine-learning model was trained to predict nightly prices and then evaluated on a **holdout set** — listings it had never seen before.

The best model was **XGBoost**. On the holdout set it achieved:

- **Average error (MAE): $54** — a typical prediction is off by about this much.
- **RMSE: $107** — like MAE, but weighs large errors more heavily.
- **MAPE: 38%** — the average error as a share of the actual price.
- **R²: 0.50** — the share of price variation the model explains (0–1).
- **50%** of predictions were within +/-25% of the true price.


## How to read this report

- **MAE (mean absolute error)** — the average difference between predicted and actual prices, in dollars. Lower is better; it's the easiest number to grasp ("off by about $X on average").
- **RMSE (root mean squared error)** — similar to MAE but penalizes large mistakes more. If RMSE is much higher than MAE, a few big errors are inflating it.
- **MAPE (mean absolute percentage error)** — the same idea as MAE, but expressed as a percentage of the actual price, so it can be compared across cities.
- **R²** — how much of the variation in prices the model explains (0 = no better than guessing the average; 1 = perfect).
- **Holdout set** — the 10% of listings the model never saw while training. All accuracy numbers come from this set, so they are an honest estimate of real-world performance.
- **Top features** — the listing attributes the model leans on most when setting a price.

## Data

- Source: [InsideAirbnb](https://insideairbnb.com/get-the-data/) (latest available snapshot)
- Listings after cleaning: **20,453** — median nightly price **$108**, mean **$152**
- Features used by the model: **26**
- Train / validation / holdout split: **14,317 / 3,681 / 2,455** listings (70/15/10, random seed 42)

## Market overview

### Price distribution
![price_dist](artifacts/figures/milan/price_dist.png)
- Median nightly price: **$108**; the middle 50% of listings fall between **$80** and **$160**.
- The mean (**$152**) sits well above the median — a minority of expensive listings pull the average up, so the **median is the better guide** to a typical price.

### Price by room type
![price_by_room_type](artifacts/figures/milan/price_by_room_type.png)

| Room type | Listings | Median price | Mean price |
|---|---|---|---|
| Hotel room | 3 | $190 | $182 |
| Entire home/apt | 18,056 | $112 | $159 |
| Private room | 2,335 | $70 | $100 |
| Shared room | 59 | $47 | $58 |


### Price by number of bedrooms
![price_by_bedrooms](artifacts/figures/milan/price_by_bedrooms.png)

### Feature correlations
![corr_heatmap](artifacts/figures/milan/corr_heatmap.png)

## Model accuracy on the holdout set

Metrics for the chosen model (**XGBoost**) on 2,455 unseen listings:

| Metric | Value | What it means |
|---|---|---|
| MAE | $54 | Average prediction is off by this much |
| RMSE | $107 | Errors, with big misses weighted more |
| MAPE | 38% | Average error as a % of the true price |
| R² | 0.50 | Share of price variation explained |

**Accuracy in practice:** 21% of predictions were within +/-10% of the true price, 50% within +/-25%, and 76% within +/-50%.

![pred_vs_actual](artifacts/milan/pred_vs_actual.png)
*Points close to the dashed line are accurate predictions; points above it mean the model under-predicted.*

![residuals](artifacts/milan/residuals.png)
*Residual = actual - predicted. A cloud centered on the zero line means errors are balanced rather than systematically too high or too low.*


## Models & results

| Model | Validation RMSE |
|---|---|
| Random Forest | $121 |
| XGBoost | $112 |
| TabPFN (capped sample, not directly comparable) | $108 |

The model with the lowest validation RMSE was then scored once on the holdout set — see the accuracy section above for its final numbers.

## Top features

The listing attributes that influence price the most (importance = how much the model relies on them):

| Feature | Importance | Meaning |
|---|---|---|
| room_type_Entire home/apt | 0.053 | the room type (entire home vs private/shared room) (specifically Entire home/apt) |
| room_type_Private room | 0.052 | the room type (entire home vs private/shared room) (specifically Private room) |
| bedrooms | 0.045 | the number of bedrooms |
| bathrooms | 0.044 | the number of bathrooms |
| instant_bookable | 0.030 | whether the listing is instantly bookable |
| host_response_time_nan | 0.030 | how quickly the host responds (specifically nan) |
| accommodates | 0.026 | how many guests a listing sleeps |
| host_neighbourhood_South Kensington | 0.026 | the host's neighbourhood (specifically South Kensington) |
| host_listings_count | 0.025 | how many listings the host manages |
| host_total_listings_count | 0.025 | a key listing attribute |


![shap_summary](artifacts/milan/shap_summary.png)
*SHAP summary: each dot is one listing; colour shows the feature's value (red = high, blue = low) and position shows whether it pushed the predicted price up (right) or down (left).*

## Prediction examples

A few real holdout listings with the model's prediction:

| Listing | Actual | Predicted | Error | Error % |
|---|---|---|---|---|
| accommodates=2, bedrooms=1, room_type=Private room | $155 | $160 | $5 | 3.5% |
| accommodates=5, bedrooms=2, room_type=Entire home/apt | $127 | $151 | $24 | 18.7% |
| accommodates=3, bedrooms=1, room_type=Entire home/apt | $78 | $124 | $46 | 59.6% |
| accommodates=2, bedrooms=1, room_type=Entire home/apt | $97 | $137 | $40 | 41.1% |
| accommodates=2, bedrooms=1, room_type=Entire home/apt | $1,009 | $1,016 | $7 | 0.7% |
| accommodates=2, bedrooms=1, room_type=Entire home/apt | $76 | $87 | $11 | 14.9% |
| accommodates=2, bedrooms=1, room_type=Entire home/apt | $131 | $135 | $4 | 2.9% |
| accommodates=4, bedrooms=1, room_type=Entire home/apt | $89 | $88 | $1 | 1.0% |


## Recommendations


- The best-performing model was **XGBoost** (holdout RMSE $107, MAE $54, R² 0.50).

- On validation, **XGBoost** beat Random Forest by 7% on RMSE ($112 vs $121). A simple ensemble or blending the two may squeeze out further gains.

- Pricing is most strongly driven by **room_type_Entire home/apt, room_type_Private room, bedrooms** — i.e. room_type_Entire home/apt (the room type (entire home vs private/shared room) (specifically Entire home/apt)); room_type_Private room (the room type (entire home vs private/shared room) (specifically Private room)); bedrooms (the number of bedrooms). Hosts can use these levers when setting nightly rates.

- On average, predictions are off by about **38%** of the actual price (median error 25%), and **50%** of predictions landed within +/-25% of the true price.

- Accuracy caveat: the holdout RMSE of $107 is **99% of the median listing price ($108)**, so predictions for typical listings are expected to be off by roughly +/-$107.

- Next steps: add richer features (neighbourhood_cleansed, amenities, review scores breakdowns), train on a log-transformed target to tame high-end outliers, run a deeper hyperparameter search, and extend the pipeline to more cities.


## Limitations

- The model only "sees" the features listed in the config — it cannot capture neighbourhood charm, amenities, photos, or seasonality.
- Extreme high-end prices are clipped at the 99th percentile during cleaning, so predictions at the very top of the market are less meaningful.
- The accuracy numbers are averages: individual predictions can be much better or much worse.
- This report is generated automatically from the pipeline artifacts and reflects the data as of 2026-08-25.

---
*Automated pipeline: fetch → preprocess → EDA → train → evaluate → report. Code in `src/`, config in `configs/base.yaml`.*