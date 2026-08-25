# src/agent.py
import argparse, subprocess, os, datetime, math
import pandas as pd
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
}


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
        best = h.get("chosen_model", "unknown")
        recs.append(
            f"The best-performing model was **{best}** (holdout RMSE ${h['rmse']:.0f}, "
            f"MAE ${h['mae']:.0f}, R² {h['r2']:.2f})."
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

    if h and ctx.get("median_price"):
        pct = h["rmse"] / ctx["median_price"] * 100
        recs.append(
            f"Accuracy caveat: the holdout RMSE of ${h['rmse']:.0f} is **{pct:.0f}% of the median "
            f"listing price (${ctx['median_price']:.0f})**, so predictions for typical listings are "
            f"expected to be off by roughly ±${h['rmse']:.0f}."
        )

    recs.append(
        "Next steps: add richer features (neighbourhood_cleansed, amenities, review scores breakdowns), "
        "train on a log-transformed target to tame high-end outliers, run a deeper hyperparameter search, "
        "and extend the pipeline to more cities."
    )
    return recs


def generate_report(city, cfg):
    city_title = city.replace("-", " ").title()
    artifact_dir = os.path.join(cfg["output"]["artifacts_dir"], city)
    figures_dir = os.path.join(cfg["output"]["artifacts_dir"], "figures", city)

    ctx = {
        "city": city,
        "city_title": city_title,
        "generated_at": datetime.date.today().isoformat(),
        "source_url": "https://insideairbnb.com/get-the-data/",
    }

    # --- stats / data overview ---
    stats = read_csv_safe(os.path.join(figures_dir, "stats.csv"))
    if stats is not None and "price" in stats.index:
        ctx["num_rows"] = int(stats.loc["price", "count"])
        ctx["median_price"] = float(stats.loc["price", "50%"])
        ctx["mean_price"] = float(stats.loc["price", "mean"])

    # --- holdout results ---
    holdout = read_csv_safe(os.path.join(artifact_dir, "holdout_results.csv"))
    if holdout is not None:
        ctx["holdout"] = {k: v for k, v in holdout[holdout.columns[0]].items()}
        for k in ("rmse", "mae", "r2"):
            ctx["holdout"][k] = to_float(ctx["holdout"].get(k))

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

    ctx["recommendations"] = build_recommendations(ctx)

    # --- template (plain Jinja, NOT an f-string — the {{ }} must survive to render) ---
    template_md = """
# Airbnb Price Prediction Report - {{ city_title }}

*Generated on {{ generated_at }} by the automated pipeline (`make all`).*

## Executive summary
{% if holdout %}
The pipeline trained and compared Random Forest, XGBoost{% if val and val.tabpfn is not none %} and TabPFN{% endif %}
on Airbnb listing data for **{{ city_title }}** ({{ num_rows | default('N/A', true) }} listings).
The **{{ holdout.chosen_model }}** model was selected for final evaluation on a held-out test set and achieved:
- **RMSE: ${{ '%.0f' | format(holdout.rmse) }}**
- **MAE: ${{ '%.0f' | format(holdout.mae) }}**
- **R²: {{ '%.2f' | format(holdout.r2) }}**
{% else %}
No evaluation results found — run `make eval` first.
{% endif %}

## Data
- Source: InsideAirbnb (latest available snapshot). See: {{ source_url }}
- City: {{ city_title }} · {{ num_rows | default('N/A', true) }} listings after cleaning

## EDA
![price_dist](artifacts/figures/{{ city }}/price_dist.png)
{% if median_price %}
- Median price: **${{ '%.0f' | format(median_price) }}** (mean ${{ '%.0f' | format(mean_price) }})
{% endif %}
![price_by_room_type](artifacts/figures/{{ city }}/price_by_room_type.png)
![corr_heatmap](artifacts/figures/{{ city }}/corr_heatmap.png)

## Models & Results

| Model | Validation RMSE |
|---|---|
{% if val and val.rf is not none %}| Random Forest | ${{ '%.0f' | format(val.rf) }} |{% endif %}
{% if val and val.xgb is not none %}| XGBoost | ${{ '%.0f' | format(val.xgb) }} |{% endif %}
{% if val and val.tabpfn is not none %}| TabPFN (capped sample, not directly comparable) | ${{ '%.0f' | format(val.tabpfn) }} |{% endif %}

Holdout results (best model): see `artifacts/{{ city }}/holdout_results.csv`.

## Top features
{% if top_features %}
| Feature | Importance |
|---|---|
{% for name, imp in top_features %}| {{ name }} | {{ '%.3f' | format(imp) }} |
{% endfor %}
{% else %}
Feature importances not available for the chosen model.
{% endif %}

## Recommendations
{% for rec in recommendations %}
- {{ rec }}
{% endfor %}

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
    except Exception:
        print("Pandoc not available — report left as markdown:", md_path)


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
