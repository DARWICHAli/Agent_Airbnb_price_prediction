# src/agent.py
import argparse, subprocess, os, datetime, math
import pandas as pd
import numpy as np
import yaml
from jinja2 import Template


def to_float(v):
    """Coerce a CSV cell to float, returning None for missing/NaN values."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


FEATURE_GLOSSARY = {
    "accommodates": "how many guests a listing sleeps",
    "bedrooms": "the number of bedrooms",
    "beds": "the number of beds",
    "room_type": "the room type (entire home vs private/shared room)",
    "property_type": "the property type (apartment, house, etc.)",
    "latitude": "the listing's location (latitude)",
    "longitude": "the listing's location (longitude)",
    "host_neighbourhood": "the host's neighbourhood",
    "review_scores_rating": "the overall review rating",
    "number_of_reviews": "the number of reviews",
    "reviews_per_month": "the review volume per month",
    "host_is_superhost": "whether the host is a superhost",
    "host_response_time": "how quickly the host responds",
    "host_response_rate": "the host's response rate",
    "host_acceptance_rate": "the host's acceptance rate",
    "host_listings_count": "how many listings the host manages",
    "availability_30": "availability over the next 30 days",
    "availability_60": "availability over the next 60 days",
    "availability_90": "availability over the next 90 days",
    "availability_365": "availability over the next 365 days",
    "minimum_nights": "the minimum nights required",
    "maximum_nights": "the maximum nights allowed",
    "instant_bookable": "whether the listing is instantly bookable",
    "host_months_since": "how long the host has been hosting (months)",
    "bathrooms": "the number of bathrooms",
}

MODEL_DISPLAY = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "tabpfn": "TabPFN",
}


def model_display(name):
    return MODEL_DISPLAY.get(name, name)


def load_config(path="configs/base.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def read_csv_safe(path, index_col=0):
    try:
        return pd.read_csv(path, index_col=index_col)
    except Exception as e:
        print("Could not read", path, "-", e)
        return None


def feature_gloss(name):
    """Human-readable description for raw or one-hot-encoded feature names."""
    if name in FEATURE_GLOSSARY:
        return FEATURE_GLOSSARY[name]
    # match encoded names like "host_neighbourhood_New Town" back to the base feature
    for key in sorted(FEATURE_GLOSSARY, key=len, reverse=True):
        if name.startswith(key + "_"):
            value = name[len(key) + 1:]
            return f"{FEATURE_GLOSSARY[key]} (specifically {value})"
    return "a key listing attribute"


def build_recommendations(ctx):
    """Rule-based 'agent' logic: turn the artifacts into actionable recommendations."""
    recs = []
    h = ctx.get("holdout")
    val = ctx.get("val") or {}
    top = ctx.get("top_features") or []

    if h:
        recs.append(
            f"The best-performing model was **{model_display(h['chosen_model'])}** "
            f"(holdout RMSE ${h['rmse']:.0f}, MAE ${h['mae']:.0f}, R² {h['r2']:.2f})."
        )

    if val.get("rf") and val.get("xgb"):
        rf, xgb = val["rf"], val["xgb"]
        if rf and xgb:
            winner, loser = ("Random Forest", "XGBoost") if rf <= xgb else ("XGBoost", "Random Forest")
            diff = abs(rf - xgb) / max(rf, xgb) * 100
            recs.append(
                f"On validation, **{winner}** beat {loser} by {diff:.0f}% on RMSE "
                f"(${min(rf, xgb):.0f} vs ${max(rf, xgb):.0f}). A simple ensemble or blending the two "
                f"may squeeze out further gains."
            )

    if top:
        names = [n for n, _ in top[:3]]
        descs = "; ".join(f"{n} ({feature_gloss(n)})" for n in names)
        recs.append(
            f"Pricing is most strongly driven by **{', '.join(names)}** — i.e. {descs}. "
            f"Hosts can use these levers when setting nightly rates."
        )

    if h and h.get("mape_pct") is not None:
        recs.append(
            f"On average, predictions are off by about **{h['mape_pct']:.0f}%** of the actual price "
            f"(median error {h['mdape_pct']:.0f}%), and **{h['within_25_pct']:.0f}%** of predictions "
            f"landed within +/-25% of the true price."
        )

    if h and ctx.get("median_price"):
        pct = h["rmse"] / ctx["median_price"] * 100
        recs.append(
            f"Accuracy caveat: the holdout RMSE of ${h['rmse']:.0f} is **{pct:.0f}% of the median "
            f"listing price (${ctx['median_price']:.0f})**, so predictions for typical listings are "
            f"expected to be off by roughly +/-${h['rmse']:.0f}."
        )

    recs.append(
        "Next steps: add richer features (neighbourhood_cleansed, amenities, review scores breakdowns), "
        "train on a log-transformed target to tame high-end outliers, run a deeper hyperparameter search, "
        "and extend the pipeline to more cities."
    )
    return recs


def fmt_int(n):
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "N/A"


def generate_report(city, cfg):
    city_title = city.replace("-", " ").title()
    artifact_dir = os.path.join(cfg["output"]["artifacts_dir"], city)
    figures_dir = os.path.join(cfg["output"]["artifacts_dir"], "figures", city)
    proc_dir = os.path.join(cfg["data"]["processed_dir"], city)

    ctx = {
        "city": city,
        "city_title": city_title,
        "generated_at": datetime.date.today().isoformat(),
        "source_url": "https://insideairbnb.com/get-the-data/",
        "feature_gloss": feature_gloss,
        "model_display": model_display,
    }

    # --- data overview (full cleaned dataset) ---
    stats = read_csv_safe(os.path.join(figures_dir, "stats.csv"))
    if stats is not None and "price" in stats.index:

        def stat(k, default=0.0):
            try:
                return float(stats.loc["price", k])
            except Exception:
                return default

        ctx["num_rows"] = int(stat("count"))
        ctx["num_rows_fmt"] = fmt_int(ctx["num_rows"])
        ctx["median_price"] = stat("50%")
        ctx["mean_price"] = stat("mean")
        ctx["p25_price"] = stat("25%")
        ctx["p75_price"] = stat("75%")

    # --- split sizes + feature count ---
    ctx["splits"] = []
    n_features = None
    for key in ("train", "val", "test"):
        p = os.path.join(proc_dir, f"{key}.parquet")
        if os.path.exists(p):
            df = pd.read_parquet(p)
            ctx["splits"].append({"key": key, "n": len(df), "n_fmt": fmt_int(len(df))})
            if n_features is None:
                n_features = max(df.shape[1] - 1, 0)  # exclude the target column
    ctx["n_features"] = n_features

    # --- holdout results (chosen model) ---
    holdout = read_csv_safe(os.path.join(artifact_dir, "holdout_results.csv"))
    if holdout is not None:
        h = {k: v for k, v in holdout[holdout.columns[0]].items()}
        for k in ("rmse", "mae", "r2", "mape_pct", "mdape_pct",
                  "within_10_pct", "within_25_pct", "within_50_pct", "n_holdout",
                  "holdout_median_price"):
            h[k] = to_float(h.get(k))
        h["n_holdout_fmt"] = fmt_int(h.get("n_holdout"))
        ctx["holdout"] = h

    # --- validation summary ---
    summary = read_csv_safe(os.path.join(artifact_dir, "model_summary.csv"))
    if summary is not None:
        vals = {k: v for k, v in summary[summary.columns[0]].items()}
        ctx["val"] = {
            "rf": to_float(vals.get("rf_rmse")),
            "xgb": to_float(vals.get("xgb_rmse")),
            "tabpfn": to_float(vals.get("tabpfn_rmse")),
        }

    # --- feature importances ---
    fi = read_csv_safe(os.path.join(artifact_dir, "feature_importances.csv"))
    if fi is not None and len(fi) > 0:
        col = fi.columns[0]
        fi_sorted = fi[col].dropna().sort_values(ascending=False)
        ctx["top_features"] = [(str(idx), float(v)) for idx, v in fi_sorted.head(10).items()]

    # --- price by room type ---
    rt = read_csv_safe(os.path.join(figures_dir, "room_type_stats.csv"))
    ctx["room_type_stats"] = []
    if rt is not None and len(rt) > 0:
        for room, row in rt.iterrows():
            ctx["room_type_stats"].append({
                "room": str(room),
                "count_fmt": fmt_int(row.get("count")),
                "median": float(row.get("median", 0.0)),
                "mean": float(row.get("mean", 0.0)),
            })

    # --- prediction examples ---
    preds_df = read_csv_safe(os.path.join(artifact_dir, "predictions_sample.csv"))
    ctx["predictions"] = []
    if preds_df is not None and len(preds_df) > 0:
        for _, r in preds_df.iterrows():
            def fmt_cell(c):
                v = r.get(c)
                if v is None:
                    return "-"
                try:
                    f = float(v)
                    if math.isnan(f):
                        return "-"
                    return str(int(f)) if f.is_integer() else f"{f:g}"
                except (TypeError, ValueError):
                    return str(v)

            desc_parts = []
            for c in ("accommodates", "bedrooms", "room_type"):
                if c in preds_df.columns:
                    desc_parts.append(f"{c}={fmt_cell(c)}")
            ctx["predictions"].append({
                "desc": ", ".join(desc_parts) or "listing",
                "actual": f"{float(r['actual_price']):,.0f}",
                "predicted": f"{float(r['predicted_price']):,.0f}",
                "error": f"{float(r['abs_error']):,.0f}",
                "pct": f"{float(r['pct_error']):.1f}%",
            })

    ctx["recommendations"] = build_recommendations(ctx)

    # --- template (plain Jinja, NOT an f-string — the {{ }} must survive to render) ---
    template_md = """
# Airbnb Price Prediction Report — {{ city_title }}

*Generated on {{ generated_at }} by the automated pipeline (`make all`). Source: [InsideAirbnb]({{ source_url }}).*

## Executive summary

This report analyzes the **{{ city_title }}** Airbnb market{% if num_rows %} using {{ num_rows_fmt }} cleaned listings{% endif %} and {{ n_features | default('N/A', true) }} listing features. A machine-learning model was trained to predict nightly prices and then evaluated on a **holdout set** — listings it had never seen before.

{% if holdout %}The best model was **{{ model_display(holdout.chosen_model) }}**. On the holdout set it achieved:

- **Average error (MAE): ${{ '%.0f' | format(holdout.mae) }}** — a typical prediction is off by about this much.
- **RMSE: ${{ '%.0f' | format(holdout.rmse) }}** — like MAE, but weighs large errors more heavily.
- **MAPE: {{ '%.0f' | format(holdout.mape_pct) }}%** — the average error as a share of the actual price.
- **R²: {{ '%.2f' | format(holdout.r2) }}** — the share of price variation the model explains (0–1).
- **{{ '%.0f' | format(holdout.within_25_pct) }}%** of predictions were within +/-25% of the true price.
{% else %}No evaluation results found — run `make eval` first.
{% endif %}

## How to read this report

- **MAE (mean absolute error)** — the average difference between predicted and actual prices, in dollars. Lower is better; it's the easiest number to grasp ("off by about $X on average").
- **RMSE (root mean squared error)** — similar to MAE but penalizes large mistakes more. If RMSE is much higher than MAE, a few big errors are inflating it.
- **MAPE (mean absolute percentage error)** — the same idea as MAE, but expressed as a percentage of the actual price, so it can be compared across cities.
- **R²** — how much of the variation in prices the model explains (0 = no better than guessing the average; 1 = perfect).
- **Holdout set** — the 10% of listings the model never saw while training. All accuracy numbers come from this set, so they are an honest estimate of real-world performance.
- **Top features** — the listing attributes the model leans on most when setting a price.

## Data

- Source: [InsideAirbnb]({{ source_url }}) (latest available snapshot)
- Listings after cleaning: **{{ num_rows_fmt | default('N/A', true) }}**{% if median_price %} — median nightly price **${{ '%.0f' | format(median_price) }}**, mean **${{ '%.0f' | format(mean_price) }}**{% endif %}
- Features used by the model: **{{ n_features | default('N/A', true) }}**
- Train / validation / holdout split:{% if splits %} **{{ splits[0].n_fmt }} / {{ splits[1].n_fmt }} / {{ splits[2].n_fmt }}** listings (70/15/10, random seed 42){% else %} N/A{% endif %}

## Market overview

### Price distribution
![price_dist](artifacts/figures/{{ city }}/price_dist.png)
{% if median_price %}- Median nightly price: **${{ '%.0f' | format(median_price) }}**; the middle 50% of listings fall between **${{ '%.0f' | format(p25_price) }}** and **${{ '%.0f' | format(p75_price) }}**.
- The mean (**${{ '%.0f' | format(mean_price) }}**) sits well above the median — a minority of expensive listings pull the average up, so the **median is the better guide** to a typical price.{% endif %}

### Price by room type
![price_by_room_type](artifacts/figures/{{ city }}/price_by_room_type.png)

{% if room_type_stats %}| Room type | Listings | Median price | Mean price |
|---|---|---|---|
{% for rt in room_type_stats %}| {{ rt.room }} | {{ rt.count_fmt }} | ${{ '%.0f' | format(rt.median) }} | ${{ '%.0f' | format(rt.mean) }} |
{% endfor %}{% endif %}

### Price by number of bedrooms
![price_by_bedrooms](artifacts/figures/{{ city }}/price_by_bedrooms.png)

### Feature correlations
![corr_heatmap](artifacts/figures/{{ city }}/corr_heatmap.png)

## Model accuracy on the holdout set

{% if holdout %}Metrics for the chosen model (**{{ model_display(holdout.chosen_model) }}**) on {{ holdout.n_holdout_fmt }} unseen listings:

| Metric | Value | What it means |
|---|---|---|
| MAE | ${{ '%.0f' | format(holdout.mae) }} | Average prediction is off by this much |
| RMSE | ${{ '%.0f' | format(holdout.rmse) }} | Errors, with big misses weighted more |
| MAPE | {{ '%.0f' | format(holdout.mape_pct) }}% | Average error as a % of the true price |
| R² | {{ '%.2f' | format(holdout.r2) }} | Share of price variation explained |

**Accuracy in practice:** {{ '%.0f' | format(holdout.within_10_pct) }}% of predictions were within +/-10% of the true price, {{ '%.0f' | format(holdout.within_25_pct) }}% within +/-25%, and {{ '%.0f' | format(holdout.within_50_pct) }}% within +/-50%.

![pred_vs_actual](artifacts/{{ city }}/pred_vs_actual.png)
*Points close to the dashed line are accurate predictions; points above it mean the model under-predicted.*

![residuals](artifacts/{{ city }}/residuals.png)
*Residual = actual - predicted. A cloud centered on the zero line means errors are balanced rather than systematically too high or too low.*
{% else %}Run `make eval` first.{% endif %}

## Models & results

| Model | Validation RMSE |
|---|---|
{% if val and val.rf is not none %}| Random Forest | ${{ '%.0f' | format(val.rf) }} |{% endif %}
{% if val and val.xgb is not none %}| XGBoost | ${{ '%.0f' | format(val.xgb) }} |{% endif %}
{% if val and val.tabpfn is not none %}| TabPFN (capped sample, not directly comparable) | ${{ '%.0f' | format(val.tabpfn) }} |{% endif %}

The model with the lowest validation RMSE was then scored once on the holdout set — see the accuracy section above for its final numbers.

## Top features

The listing attributes that influence price the most (importance = how much the model relies on them):

| Feature | Importance | Meaning |
|---|---|---|
{% for name, imp in top_features %}| {{ name }} | {{ '%.3f' | format(imp) }} | {{ feature_gloss(name) }} |
{% endfor %}

![shap_summary](artifacts/{{ city }}/shap_summary.png)
*SHAP summary: each dot is one listing; colour shows the feature's value (red = high, blue = low) and position shows whether it pushed the predicted price up (right) or down (left).*

## Prediction examples

A few real holdout listings with the model's prediction:

| Listing | Actual | Predicted | Error | Error % |
|---|---|---|---|---|
{% for row in predictions %}| {{ row.desc }} | ${{ row.actual }} | ${{ row.predicted }} | ${{ row.error }} | {{ row.pct }} |
{% endfor %}

## Recommendations

{% for rec in recommendations %}
- {{ rec }}
{% endfor %}

## Limitations

- The model only "sees" the features listed in the config — it cannot capture neighbourhood charm, amenities, photos, or seasonality.
- Extreme high-end prices are clipped at the 99th percentile during cleaning, so predictions at the very top of the market are less meaningful.
- The accuracy numbers are averages: individual predictions can be much better or much worse.
- This report is generated automatically from the pipeline artifacts and reflects the data as of {{ generated_at }}.

---
*Automated pipeline: fetch → preprocess → EDA → train → evaluate → report. Code in `src/`, config in `configs/base.yaml`.*
"""
    content = Template(template_md).render(**ctx)

    os.makedirs(cfg["output"]["reports_dir"], exist_ok=True)
    md_path = os.path.join(cfg["output"]["reports_dir"], f"{city}_report.md")
    with open(md_path, "w") as f:
        f.write(content)
    print("Report written:", md_path)

    pdf_path = os.path.join(cfg["output"]["reports_dir"], f"{city}_report.pdf")
    try:
        subprocess.check_call(["pandoc", md_path, "-o", pdf_path])
        print("PDF created:", pdf_path)
    except Exception as e:
        print("PDF generation failed (pandoc or LaTeX error):", str(e)[:300])
        print("Report left as markdown:", md_path)


def run_stage(stage, city):
    if stage in ("all", "fetch"):
        subprocess.check_call(["python", "-u", "src/fetcher.py", "--city", city])
    if stage in ("all", "preprocess"):
        subprocess.check_call(["python", "-u", "src/preprocess.py", "--city", city])
    if stage in ("all", "eda"):
        subprocess.check_call(["python", "-u", "src/eda.py", "--city", city])
    if stage in ("all", "train"):
        subprocess.check_call(["python", "-u", "src/train.py", "--city", city, "--config", "configs/base.yaml"])
    if stage in ("all", "eval"):
        subprocess.check_call(["python", "-u", "src/evaluate.py", "--city", city])
    if stage in ("all", "report", "generate_report"):
        cfg = load_config("configs/base.yaml")
        generate_report(city, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="paris")
    parser.add_argument("--stage", default="all",
                        choices=["all", "fetch", "preprocess", "eda", "train", "eval", "report", "generate_report"])
    args = parser.parse_args()
    run_stage(args.stage, args.city)
