# src/evaluate.py
import argparse, os, joblib, pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import shap
import yaml


def load_config(path="configs/base.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def pick_best_model(artifact_dir, summary):
    """Choose the model with the lowest validation RMSE among available joblib artifacts."""
    candidates = {
        "random_forest": ("rf_rmse", os.path.join(artifact_dir, "random_forest.joblib")),
        "xgboost": ("xgb_rmse", os.path.join(artifact_dir, "xgboost.joblib")),
    }
    available = {name: path for name, (key, path) in candidates.items() if os.path.exists(path)}
    if not available:
        raise SystemExit(f"No model artifacts found in {artifact_dir}. Run train first.")

    rmse_map = {}
    for name, path in available.items():
        key = candidates[name][0]
        try:
            rmse_map[name] = float(summary.loc[key, summary.columns[0]])
        except Exception:
            continue
    if rmse_map:
        chosen_name = min(rmse_map, key=rmse_map.get)
    else:
        chosen_name = list(available)[0]
    return chosen_name, available[chosen_name]


def encoded_feature_names(preproc):
    """Map encoded (post-OneHot) columns back to readable feature names."""
    names = preproc.get_feature_names_out()
    return [n.replace("num__", "").replace("cat__", "") for n in names]


def plot_pred_vs_actual(y, preds, artifact_dir, chosen_name, r2):
    lims = (min(y.min(), preds.min()), max(y.max(), preds.max()))
    plt.figure(figsize=(7, 7))
    plt.scatter(y, preds, s=12, alpha=0.35, color="#4C72B0", edgecolors="none")
    plt.plot(lims, lims, "k--", lw=1.2, label="Perfect prediction")
    plt.xlabel("Actual nightly price ($)")
    plt.ylabel("Predicted nightly price ($)")
    plt.title(f"Predicted vs actual price — {chosen_name} (holdout, R²={r2:.2f})")
    plt.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, "pred_vs_actual.png"), bbox_inches="tight", dpi=150)
    plt.close()


def plot_residuals(y, preds, artifact_dir, chosen_name, rmse):
    resid = y - preds
    plt.figure(figsize=(8, 5))
    plt.scatter(preds, resid, s=12, alpha=0.35, color="#55A868", edgecolors="none")
    plt.axhline(0, color="k", lw=1.2)
    plt.axhline(rmse, color="gray", ls=":", lw=1, label=f"+RMSE ${rmse:.0f}")
    plt.axhline(-rmse, color="gray", ls=":", lw=1, label=f"−RMSE ${rmse:.0f}")
    plt.xlabel("Predicted nightly price ($)")
    plt.ylabel("Residual (actual − predicted, $)")
    plt.title(f"Residuals vs predicted price — {chosen_name} (holdout)")
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, "residuals.png"), bbox_inches="tight", dpi=150)
    plt.close()


def save_predictions_sample(X_hold, y, preds, pct_err, artifact_dir, cfg, city):
    sample_cols = ["accommodates", "bedrooms", "bathrooms", "room_type", "property_type"]
    show_cols = [c for c in sample_cols if c in X_hold.columns]
    sample_df = X_hold[show_cols].copy()
    sample_df["actual_price"] = np.round(y, 0)
    sample_df["predicted_price"] = np.round(preds, 0)
    sample_df["abs_error"] = np.round(np.abs(y - preds), 0)
    sample_df["pct_error"] = np.round(pct_err, 1)
    sample_df = sample_df.sample(n=min(8, len(sample_df)), random_state=cfg["seed"])
    out = os.path.join(artifact_dir, "predictions_sample.csv")
    sample_df.to_csv(out)
    print("Predictions sample saved (%d rows) to %s" % (len(sample_df), out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="paris")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    proc = os.path.join(cfg["data"]["processed_dir"], args.city)
    artifact_dir = os.path.join(cfg["output"]["artifacts_dir"], args.city)

    holdout = pd.read_parquet(os.path.join(proc, "test.parquet"))
    target = cfg["training"]["target"]
    X_hold = holdout.drop(columns=[target])
    y_hold = holdout[target]

    summary = pd.read_csv(os.path.join(artifact_dir, "model_summary.csv"), index_col=0)
    chosen_name, chosen_path = pick_best_model(artifact_dir, summary)
    print("Chosen model:", chosen_name, "->", chosen_path)

    model = joblib.load(chosen_path)
    preds = np.asarray(model.predict(X_hold), dtype=float)
    y = np.asarray(y_hold, dtype=float)

    rmse = root_mean_squared_error(y, preds)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    # Percentage-error metrics — the most intuitive for a general audience.
    # Prices are always > 0 after cleaning, so relative errors are well defined.
    preds_clip = np.maximum(preds, 0.0)
    pct_err = np.abs((y - preds_clip) / y) * 100.0
    mape = float(np.mean(pct_err))
    mdape = float(np.median(pct_err))
    within_10 = float(np.mean(pct_err <= 10.0) * 100.0)
    within_25 = float(np.mean(pct_err <= 25.0) * 100.0)
    within_50 = float(np.mean(pct_err <= 50.0) * 100.0)

    res = {
        "chosen_model": chosen_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape_pct": mape,
        "mdape_pct": mdape,
        "within_10_pct": within_10,
        "within_25_pct": within_25,
        "within_50_pct": within_50,
        "n_holdout": len(y),
        "holdout_median_price": float(np.median(y)),
    }
    pd.Series(res).to_csv(os.path.join(artifact_dir, "holdout_results.csv"))
    print("Holdout results saved:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})

    # Diagnostic plots + prediction examples
    plot_pred_vs_actual(y, preds_clip, artifact_dir, chosen_name, r2)
    plot_residuals(y, preds_clip, artifact_dir, chosen_name, rmse)
    save_predictions_sample(X_hold, y, preds_clip, pct_err, artifact_dir, cfg, args.city)

    # Feature importance + SHAP for tree pipelines (skip non-pipeline models like TabPFN)
    if hasattr(model, "named_steps"):
        try:
            preproc = model.named_steps["preproc"]
            final = model.named_steps[list(model.named_steps.keys())[-1]]
            X_enc = preproc.transform(X_hold)
            feature_names = encoded_feature_names(preproc)

            if hasattr(final, "feature_importances_"):
                importances = pd.Series(final.feature_importances_, index=feature_names)
                importances.sort_values(ascending=False).to_csv(
                    os.path.join(artifact_dir, "feature_importances.csv")
                )
                print("Feature importances saved (%d features)" % len(importances))

            try:
                explainer = shap.TreeExplainer(final)
                shap_values = explainer.shap_values(X_enc)
                shap.summary_plot(shap_values, X_enc, feature_names=feature_names, show=False)
                plt.savefig(os.path.join(artifact_dir, "shap_summary.png"), bbox_inches="tight", dpi=150)
                print("SHAP summary plot saved")
            except Exception as e:
                print("SHAP failed:", str(e))
        except Exception as e:
            print("Feature importance failed:", str(e))
