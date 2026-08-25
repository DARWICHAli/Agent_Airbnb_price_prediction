
# Airbnb Price Prediction Report — Edinburgh

*Generated on 2026-08-25 by the automated pipeline (`make all`). Source: [InsideAirbnb](https://insideairbnb.com/get-the-data/).*

## Executive summary

This report analyzes the **Edinburgh** Airbnb market using 4,925 cleaned listings and 26 listing features. A machine-learning model was trained to predict nightly prices and then evaluated on a **holdout set** — listings it had never seen before.

The best model was **XGBoost**. On the holdout set it achieved:

- **Average error (MAE): $68** — a typical prediction is off by about this much.
- **RMSE: $162** — like MAE, but weighs large errors more heavily.
- **MAPE: 32%** — the average error as a share of the actual price.
- **R²: 0.64** — the share of price variation the model explains (0–1).
- **62%** of predictions were within +/-25% of the true price.


## How to read this report

- **MAE (mean absolute error)** — the average difference between predicted and actual prices, in dollars. Lower is better; it's the easiest number to grasp ("off by about $X on average").
- **RMSE (root mean squared error)** — similar to MAE but penalizes large mistakes more. If RMSE is much higher than MAE, a few big errors are inflating it.
- **MAPE (mean absolute percentage error)** — the same idea as MAE, but expressed as a percentage of the actual price, so it can be compared across cities.
- **R²** — how much of the variation in prices the model explains (0 = no better than guessing the average; 1 = perfect).
- **Holdout set** — the 10% of listings the model never saw while training. All accuracy numbers come from this set, so they are an honest estimate of real-world performance.
- **Top features** — the listing attributes the model leans on most when setting a price.

## Data

- Source: [InsideAirbnb](https://insideairbnb.com/get-the-data/) (latest available snapshot)
- Listings after cleaning: **4,925** — median nightly price **$160**, mean **$217**
- Features used by the model: **26**
- Train / validation / holdout split: **3,447 / 886 / 592** listings (70/15/10, random seed 42)

## Market overview

### Price distribution
![price_dist](artifacts/figures/edinburgh/price_dist.png)
- Median nightly price: **$160**; the middle 50% of listings fall between **$108** and **$237**.
- The mean (**$217**) sits well above the median — a minority of expensive listings pull the average up, so the **median is the better guide** to a typical price.

### Price by room type
![price_by_room_type](artifacts/figures/edinburgh/price_by_room_type.png)

| Room type | Listings | Median price | Mean price |
|---|---|---|---|
| Hotel room | 21 | $200 | $191 |
| Entire home/apt | 3,544 | $186 | $250 |
| Private room | 1,341 | $87 | $128 |
| Shared room | 19 | $68 | $286 |


### Price by number of bedrooms
![price_by_bedrooms](artifacts/figures/edinburgh/price_by_bedrooms.png)

### Feature correlations
![corr_heatmap](artifacts/figures/edinburgh/corr_heatmap.png)

## Model accuracy on the holdout set

Metrics for the chosen model (**XGBoost**) on 592 unseen listings:

| Metric | Value | What it means |
|---|---|---|
| MAE | $68 | Average prediction is off by this much |
| RMSE | $162 | Errors, with big misses weighted more |
| MAPE | 32% | Average error as a % of the true price |
| R² | 0.64 | Share of price variation explained |

**Accuracy in practice:** 29% of predictions were within +/-10% of the true price, 62% within +/-25%, and 83% within +/-50%.

![pred_vs_actual](artifacts/edinburgh/pred_vs_actual.png)
*Points close to the dashed line are accurate predictions; points above it mean the model under-predicted.*

![residuals](artifacts/edinburgh/residuals.png)
*Residual = actual - predicted. A cloud centered on the zero line means errors are balanced rather than systematically too high or too low.*


## Models & results

| Model | Validation RMSE |
|---|---|
| Random Forest | $137 |
| XGBoost | $123 |
| TabPFN (capped sample, not directly comparable) | $122 |

The model with the lowest validation RMSE was then scored once on the holdout set — see the accuracy section above for its final numbers.

## Top features

The listing attributes that influence price the most (importance = how much the model relies on them):

| Feature | Importance | Meaning |
|---|---|---|
| property_type_Private room in townhouse | 0.083 | the property type (apartment, house, etc.) (specifically Private room in townhouse) |
| host_neighbourhood_New Town | 0.078 | the host's neighbourhood (specifically New Town) |
| host_neighbourhood_Cannonmills | 0.071 | the host's neighbourhood (specifically Cannonmills) |
| bedrooms | 0.063 | the number of bedrooms |
| host_neighbourhood_Fasach | 0.057 | the host's neighbourhood (specifically Fasach) |
| host_neighbourhood_Laulerie | 0.042 | the host's neighbourhood (specifically Laulerie) |
| accommodates | 0.037 | how many guests a listing sleeps |
| availability_90 | 0.034 | availability over the next 90 days |
| bathrooms | 0.034 | the number of bathrooms |
| host_listings_count | 0.028 | how many listings the host manages |


![shap_summary](artifacts/edinburgh/shap_summary.png)
*SHAP summary: each dot is one listing; colour shows the feature's value (red = high, blue = low) and position shows whether it pushed the predicted price up (right) or down (left).*

## Prediction examples

A few real holdout listings with the model's prediction:

| Listing | Actual | Predicted | Error | Error % |
|---|---|---|---|---|
| accommodates=10, bedrooms=4, room_type=Entire home/apt | $275 | $552 | $277 | 100.8% |
| accommodates=2, bedrooms=1, room_type=Private room | $120 | $91 | $29 | 23.8% |
| accommodates=5, bedrooms=2, room_type=Entire home/apt | $228 | $234 | $6 | 2.5% |
| accommodates=2, bedrooms=1, room_type=Entire home/apt | $178 | $170 | $8 | 4.5% |
| accommodates=2, bedrooms=1, room_type=Private room | $87 | $74 | $13 | 15.4% |
| accommodates=2, bedrooms=1, room_type=Entire home/apt | $190 | $205 | $15 | 8.0% |
| accommodates=6, bedrooms=2, room_type=Entire home/apt | $206 | $211 | $5 | 2.4% |
| accommodates=2, bedrooms=1, room_type=Private room | $104 | $87 | $17 | 16.3% |


## Recommendations


- The best-performing model was **XGBoost** (holdout RMSE $162, MAE $68, R² 0.64).

- On validation, **XGBoost** beat Random Forest by 10% on RMSE ($123 vs $137). A simple ensemble or blending the two may squeeze out further gains.

- Pricing is most strongly driven by **property_type_Private room in townhouse, host_neighbourhood_New Town, host_neighbourhood_Cannonmills** — i.e. property_type_Private room in townhouse (the property type (apartment, house, etc.) (specifically Private room in townhouse)); host_neighbourhood_New Town (the host's neighbourhood (specifically New Town)); host_neighbourhood_Cannonmills (the host's neighbourhood (specifically Cannonmills)). Hosts can use these levers when setting nightly rates.

- On average, predictions are off by about **32%** of the actual price (median error 18%), and **62%** of predictions landed within +/-25% of the true price.

- Accuracy caveat: the holdout RMSE of $162 is **101% of the median listing price ($160)**, so predictions for typical listings are expected to be off by roughly +/-$162.

- Next steps: add richer features (neighbourhood_cleansed, amenities, review scores breakdowns), train on a log-transformed target to tame high-end outliers, run a deeper hyperparameter search, and extend the pipeline to more cities.


## Limitations

- The model only "sees" the features listed in the config — it cannot capture neighbourhood charm, amenities, photos, or seasonality.
- Extreme high-end prices are clipped at the 99th percentile during cleaning, so predictions at the very top of the market are less meaningful.
- The accuracy numbers are averages: individual predictions can be much better or much worse.
- This report is generated automatically from the pipeline artifacts and reflects the data as of 2026-08-25.

---
*Automated pipeline: fetch → preprocess → EDA → train → evaluate → report. Code in `src/`, config in `configs/base.yaml`.*