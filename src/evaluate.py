# src/evaluate.py
import argparse, os, joblib, pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import shap
import yaml
import numpy as np


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
    preds = model.predict(X_hold)
    rmse = root_mean_squared_error(y_hold, preds)
    mae = mean_absolute_error(y_hold, preds)
    r2 = r2_score(y_hold, preds)
    res = {"chosen_model": chosen_name, "rmse": rmse, "mae": mae, "r2": r2}
    pd.Series(res).to_csv(os.path.join(artifact_dir, "holdout_results.csv"))
    print("Holdout results saved:", res)

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
                import matplotlib.pyplot as plt
                plt.savefig(os.path.join(artifact_dir, "shap_summary.png"), bbox_inches="tight")
                print("SHAP summary plot saved")
            except Exception as e:
                print("SHAP failed:", str(e))
        except Exception as e:
            print("Feature importance failed:", str(e))
