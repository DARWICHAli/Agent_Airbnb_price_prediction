# Agent Airbnb Price Prediction

An end-to-end, agent-orchestrated pipeline that predicts Airbnb listing prices for any city. It downloads real listing data from [InsideAirbnb](https://insideairbnb.com/get-the-data/), cleans and splits it, trains several ML models, evaluates the best one on a held-out test set, and automatically generates a Markdown/PDF report with insights and recommendations.

The "agent" (`src/agent.py`) runs the whole pipeline and turns the training artifacts into a data-driven report — model comparison, top features, and actionable recommendations — with no manual analysis required.

## Pipeline

```
fetch → preprocess → eda → train → evaluate → report
```

| Stage | Script | What it does |
|---|---|---|
| `fetch` | `src/fetcher.py` | Scrapes the InsideAirbnb data page, downloads the city's `listings.csv.gz`, extracts it |
| `preprocess` | `src/preprocess.py` | Cleans prices (currency symbols, outlier clipping at the 99th percentile), derives features (`host_months_since`), imputes missing values (median/mode), splits into train/val/test (70/15/10) |
| `eda` | `src/eda.py` | Price distribution, price-by-room-type boxplot, correlation heatmap, summary stats → `artifacts/figures/<city>/` |
| `train` | `src/train.py` | Trains **Random Forest**, **XGBoost**, and **TabPFN** (optional) with a light `RandomizedSearchCV`; logs runs and metrics to MLflow (SQLite); saves `.joblib` models |
| `evaluate` | `src/evaluate.py` | Picks the best model by validation RMSE, scores it on the held-out test set, saves SHAP plot and named feature importances |
| `report` | `src/agent.py` | Renders a data-driven Markdown report (executive summary, results table, top features, recommendations) and converts it to PDF via pandoc |

## Quickstart

```bash
# 1. Set up the environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run the full pipeline for a city (edinburgh | milan | new-york-city | paris | ...)
make all city=edinburgh

# or run individual stages
make fetch city=edinburgh
make preprocess city=edinburgh
make eda city=edinburgh
make train city=edinburgh
make eval city=edinburgh
make report city=edinburgh
```

The final report is written to `reports/<city>_report.md` and `reports/<city>_report.pdf`.

## Results

Best model on validation RMSE, evaluated on the held-out test set (all models trained with the same features and a 70/15/10 split, seed 42):

| City | Listings | Median price | Best model | Val RMSE (RF / XGB / TabPFN) | Holdout RMSE | Holdout R² |
|---|---|---|---|---|---|---|
| Edinburgh | 3,447 | $160 | XGBoost | $137 / $123 / $122 | $162 | 0.64 |
| Milan | 14,317 | $108 | XGBoost | $121 / $112 / $108 | $107 | 0.50 |
| New York City | 14,929 | $154 | XGBoost | $713 / $667 / $956 | $572 | 0.92 |

Notes:
- **Price outliers are clipped** at the 99th percentile during preprocessing, which removed the extreme tails that previously inflated RMSE to ~$800.
- **TabPFN** is a transformer baseline that scales quadratically with sample size; on CPU it is fit on at most `max_train_samples` (default 4096) and evaluated on at most `max_eval_samples` (default 1024), so its RMSE is **not directly comparable** with the tree models.
- NYC has the highest RMSE but also the best R² (0.92) — the wide spread of listing prices (mean $420 vs median $154) makes absolute errors large.

## Directory structure

```
configs/base.yaml        # all pipeline configuration (features, splits, model params, tuning)
data/raw/                # downloaded listings CSVs (gitignored)
data/processed/<city>/   # train/val/test parquet files (gitignored)
artifacts/<city>/        # models (.joblib), holdout results, feature importances, SHAP plot
artifacts/figures/<city> # EDA figures + stats.csv
reports/<city>_report.*  # generated markdown + PDF reports
src/                     # pipeline scripts
mlflow.db, mlruns/       # experiment tracking (gitignored)
```

## Configuration

Everything is driven by `configs/base.yaml`:

- `data.required_columns` — feature selection; any column not in the raw CSV is skipped.
- `preprocessing.price_clip_quantile` — clips prices above this quantile (default 0.99).
- `training.*` — model parameters and whether to use TabPFN.
- `training.tuning` — `RandomizedSearchCV` settings (`n_iter`, `cv`, `max_samples`); tuning runs on a subsample of the train set and the best parameters are refit on the full train set.
- `split.*` — train/val/test fractions and seed.

## Limitations & next steps

- Reports are rule-based; they could be upgraded to an LLM-generated narrative using the same artifacts.
- A log-transformed target would likely reduce the influence of high-end outliers further.
- A deeper hyperparameter search (or Optuna) and ensembling (RF + XGBoost blend) are natural next improvements.
- New cities require only a `--city` argument, as long as InsideAirbnb hosts their data.

## License

See [LICENSE](LICENSE).
