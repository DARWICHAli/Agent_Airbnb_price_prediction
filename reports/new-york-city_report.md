
# Airbnb Price Prediction Report — New York City

*Generated on 2026-08-25 by the automated pipeline (`make all`). Source: [InsideAirbnb](https://insideairbnb.com/get-the-data/).*

## Executive summary

This report analyzes the **New York City** Airbnb market using 21,328 cleaned listings and 26 listing features. A machine-learning model was trained to predict nightly prices and then evaluated on a **holdout set** — listings it had never seen before.

The best model was **XGBoost**. On the holdout set it achieved:

- **Average error (MAE): $116** — a typical prediction is off by about this much.
- **RMSE: $572** — like MAE, but weighs large errors more heavily.
- **MAPE: 48%** — the average error as a share of the actual price.
- **R²: 0.92** — the share of price variation the model explains (0–1).
- **44%** of predictions were within +/-25% of the true price.


## How to read this report

- **MAE (mean absolute error)** — the average difference between predicted and actual prices, in dollars. Lower is better; it's the easiest number to grasp ("off by about $X on average").
- **RMSE (root mean squared error)** — similar to MAE but penalizes large mistakes more. If RMSE is much higher than MAE, a few big errors are inflating it.
- **MAPE (mean absolute percentage error)** — the same idea as MAE, but expressed as a percentage of the actual price, so it can be compared across cities.
- **R²** — how much of the variation in prices the model explains (0 = no better than guessing the average; 1 = perfect).
- **Holdout set** — the 10% of listings the model never saw while training. All accuracy numbers come from this set, so they are an honest estimate of real-world performance.
- **Top features** — the listing attributes the model leans on most when setting a price.

## Data

- Source: [InsideAirbnb](https://insideairbnb.com/get-the-data/) (latest available snapshot)
- Listings after cleaning: **21,328** — median nightly price **$154**, mean **$430**
- Features used by the model: **26**
- Train / validation / holdout split: **14,929 / 3,839 / 2,560** listings (70/15/10, random seed 42)

## Market overview

### Price distribution
![price_dist](artifacts/figures/new-york-city/price_dist.png)
- Median nightly price: **$154**; the middle 50% of listings fall between **$89** and **$279**.
- The mean (**$430**) sits well above the median — a minority of expensive listings pull the average up, so the **median is the better guide** to a typical price.

### Price by room type
![price_by_room_type](artifacts/figures/new-york-city/price_by_room_type.png)

| Room type | Listings | Median price | Mean price |
|---|---|---|---|
| Hotel room | 234 | $19514 | $15668 |
| Entire home/apt | 11,985 | $210 | $304 |
| Private room | 8,930 | $86 | $206 |
| Shared room | 179 | $67 | $141 |


### Price by number of bedrooms
![price_by_bedrooms](artifacts/figures/new-york-city/price_by_bedrooms.png)

### Feature correlations
![corr_heatmap](artifacts/figures/new-york-city/corr_heatmap.png)

## Model accuracy on the holdout set

Metrics for the chosen model (**XGBoost**) on 2,560 unseen listings:

| Metric | Value | What it means |
|---|---|---|
| MAE | $116 | Average prediction is off by this much |
| RMSE | $572 | Errors, with big misses weighted more |
| MAPE | 48% | Average error as a % of the true price |
| R² | 0.92 | Share of price variation explained |

**Accuracy in practice:** 20% of predictions were within +/-10% of the true price, 44% within +/-25%, and 70% within +/-50%.

![pred_vs_actual](artifacts/new-york-city/pred_vs_actual.png)
*Points close to the dashed line are accurate predictions; points above it mean the model under-predicted.*

![residuals](artifacts/new-york-city/residuals.png)
*Residual = actual - predicted. A cloud centered on the zero line means errors are balanced rather than systematically too high or too low.*


## Models & results

| Model | Validation RMSE |
|---|---|
| Random Forest | $713 |
| XGBoost | $667 |
| TabPFN (capped sample, not directly comparable) | $956 |

The model with the lowest validation RMSE was then scored once on the holdout set — see the accuracy section above for its final numbers.

## Top features

The listing attributes that influence price the most (importance = how much the model relies on them):

| Feature | Importance | Meaning |
|---|---|---|
| room_type_Hotel room | 0.400 | the room type (entire home vs private/shared room) (specifically Hotel room) |
| host_response_time_within an hour | 0.134 | how quickly the host responds (specifically within an hour) |
| host_neighbourhood_Gateway District | 0.084 | the host's neighbourhood (specifically Gateway District) |
| host_neighbourhood_Central Business District | 0.049 | the host's neighbourhood (specifically Central Business District) |
| host_months_since | 0.027 | how long the host has been hosting (months) |
| host_neighbourhood_Airport North | 0.025 | the host's neighbourhood (specifically Airport North) |
| property_type_Room in hotel | 0.022 | the property type (apartment, house, etc.) (specifically Room in hotel) |
| beds | 0.021 | the number of beds |
| host_neighbourhood_Clearwater Beach | 0.019 | the host's neighbourhood (specifically Clearwater Beach) |
| minimum_nights | 0.018 | the minimum nights required |


![shap_summary](artifacts/new-york-city/shap_summary.png)
*SHAP summary: each dot is one listing; colour shows the feature's value (red = high, blue = low) and position shows whether it pushed the predicted price up (right) or down (left).*

## Prediction examples

A few real holdout listings with the model's prediction:

| Listing | Actual | Predicted | Error | Error % |
|---|---|---|---|---|
| accommodates=1, bedrooms=1, room_type=Private room | $77 | $167 | $90 | 116.9% |
| accommodates=1, bedrooms=1, room_type=Private room | $117 | $118 | $1 | 0.9% |
| accommodates=8, bedrooms=4, room_type=Entire home/apt | $347 | $402 | $55 | 15.8% |
| accommodates=1, bedrooms=0, room_type=Entire home/apt | $241 | $199 | $42 | 17.4% |
| accommodates=1, bedrooms=1, room_type=Private room | $44 | $6 | $38 | 86.3% |
| accommodates=2, bedrooms=1, room_type=Entire home/apt | $212 | $235 | $23 | 10.7% |
| accommodates=2, bedrooms=1, room_type=Private room | $116 | $120 | $4 | 3.3% |
| accommodates=2, bedrooms=1, room_type=Private room | $85 | $80 | $5 | 6.4% |


## Recommendations


- The best-performing model was **XGBoost** (holdout RMSE $572, MAE $116, R² 0.92).

- On validation, **XGBoost** beat Random Forest by 6% on RMSE ($667 vs $713). A simple ensemble or blending the two may squeeze out further gains.

- Pricing is most strongly driven by **room_type_Hotel room, host_response_time_within an hour, host_neighbourhood_Gateway District** — i.e. room_type_Hotel room (the room type (entire home vs private/shared room) (specifically Hotel room)); host_response_time_within an hour (how quickly the host responds (specifically within an hour)); host_neighbourhood_Gateway District (the host's neighbourhood (specifically Gateway District)). Hosts can use these levers when setting nightly rates.

- On average, predictions are off by about **48%** of the actual price (median error 30%), and **44%** of predictions landed within +/-25% of the true price.

- Accuracy caveat: the holdout RMSE of $572 is **371% of the median listing price ($154)**, so predictions for typical listings are expected to be off by roughly +/-$572.

- Next steps: add richer features (neighbourhood_cleansed, amenities, review scores breakdowns), train on a log-transformed target to tame high-end outliers, run a deeper hyperparameter search, and extend the pipeline to more cities.


## Limitations

- The model only "sees" the features listed in the config — it cannot capture neighbourhood charm, amenities, photos, or seasonality.
- Extreme high-end prices are clipped at the 99th percentile during cleaning, so predictions at the very top of the market are less meaningful.
- The accuracy numbers are averages: individual predictions can be much better or much worse.
- This report is generated automatically from the pipeline artifacts and reflects the data as of 2026-08-25.

---
*Automated pipeline: fetch → preprocess → EDA → train → evaluate → report. Code in `src/`, config in `configs/base.yaml`.*