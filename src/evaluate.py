# src/evaluate.py
import argparse, os, joblib, pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score,root_mean_squared_error
import shap
import yaml
import numpy as np

def load_config(path="configs/base.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="paris")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    proc = os.path.join(cfg["data"]["processed_dir"], args.city)

    holdout = pd.read_parquet(os.path.join(proc, "test.parquet"))
    target = cfg["training"]["target"]
    X_hold = holdout.drop(columns=[target])
    y_hold = holdout[target]

    # choose best model by reading artifacts summary
    artifact_dir = os.path.join(cfg["output"]["artifacts_dir"],args.city)
    summary = pd.read_csv(os.path.join(artifact_dir, "model_summary.csv"), index_col=0)
    #summary = pd.read_csv(os.path.join(artifact_dir, "model_summary.csv"), index_col=0).squeeze("columns")
    # Compare rmse and choose minimal
    candidates = {
        "random_forest": os.path.join(artifact_dir, "random_forest.joblib"),
        "xgboost": os.path.join(artifact_dir, "xgboost.joblib"),
    }
    # fallback: prefer xgboost if present
    chosen = None
    # simple logic: check file exists, compare rmse in summary
    rmse_map = {}
    for k in ["rf_rmse","xgb_rmse","tabpfn_rmse"]:
        if k in summary.index:
            try:
                rmse_map[k] = float(summary.loc[k].values[0])
            except Exception:
                pass
    # pick minimal key
    min_key = min(rmse_map, key=rmse_map.get) if rmse_map else None
    if min_key and "rf" in min_key:
        chosen = os.path.join(artifact_dir, "random_forest.joblib")
    elif min_key and "xgb" in min_key:
        chosen = os.path.join(artifact_dir, "xgboost.joblib")
    else:
        # fallback
        chosen = os.path.join(artifact_dir, "xgboost.joblib")
    if not os.path.exists(chosen):
        # pick any existing artifact
        for f in os.listdir(artifact_dir):
            if f.endswith(".joblib"):
                chosen = os.path.join(artifact_dir, f)
                break
    print("Chosen model:", chosen)
    model = joblib.load(chosen)
    preds = model.predict(X_hold)
    rmse = root_mean_squared_error(y_hold, preds)
    mae = mean_absolute_error(y_hold, preds)
    r2 = r2_score(y_hold, preds)
    res = {"rmse": rmse, "mae": mae, "r2": r2}
    pd.Series(res).to_csv(os.path.join(artifact_dir, "holdout_results.csv"))
    print("Holdout results saved:", res)

    # Feature importance: try SHAP (slow); fallback to model.feature_importances_ if tree
    try:
        explainer = shap.Explainer(model.predict, X_hold)
        shap_values = explainer(X_hold)
        shap.summary_plot(shap_values, X_hold, show=False)
        import matplotlib.pyplot as plt
        plt.savefig(os.path.join(artifact_dir, "shap_summary.png"), bbox_inches="tight")
        print("SHAP plot saved")
    except Exception as e:
        print("SHAP failed:", str(e))
        try:
            importances = model.named_steps[list(model.named_steps.keys())[-1]].feature_importances_
            # can't map back to column names easily after OneHot; this is a best-effort
            pd.Series(importances).sort_values(ascending=False).head(20).to_csv(os.path.join(artifact_dir,"feature_importances.csv"))
        except Exception as ee:
            print("Feature importance failed:", ee)
