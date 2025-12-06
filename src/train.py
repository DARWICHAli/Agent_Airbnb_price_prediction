# src/train.py
import argparse, os, yaml
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import joblib
# TabPFN is optional
try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    HAS_TABPFN = True
except Exception:
    HAS_TABPFN = False

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def get_feature_sets(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return X, y, num_cols, cat_cols

def build_preprocessor(num_cols, cat_cols):
    num_pipe = StandardScaler()
    cat_pipe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    preproc = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
    return preproc

def evaluate_and_log(model, X_val, y_val, run_name, artifact_dir):
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    mlflow.log_metric("rmse", float(rmse))
    mlflow.log_metric("mae", float(mae))
    mlflow.log_metric("r2", float(r2))
    # save model
    model_path = os.path.join(artifact_dir, f"{run_name}.joblib")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)
    return {"rmse": rmse, "mae": mae, "r2": r2, "model_path": model_path}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="paris" ,type=str, required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    city = args.city
    cfg = load_config(args.config)
    proc = os.path.join(cfg["data"]["processed_dir"],city)
    train = pd.read_parquet(os.path.join(proc, f"train.parquet"))
    val = pd.read_parquet(os.path.join(proc, "val.parquet"))
    target = cfg["training"]["target"]
    X_train, y_train, num_cols, cat_cols = get_feature_sets(train, target)
    X_val, y_val, _, _ = get_feature_sets(val, target)

    preproc = build_preprocessor(num_cols, cat_cols)
    artifact_dir = os.path.join(cfg["output"]["artifacts_dir"],args.city)
    os.makedirs(artifact_dir, exist_ok=True)

    mlflow.set_tracking_uri(cfg["experiment"]["mlflow_tracking_uri"])
    mlflow.set_experiment(cfg["experiment"]["experiment_name"])

    # Random Forest
    with mlflow.start_run(run_name="random_forest"):
        rf = RandomForestRegressor(**cfg["training"]["random_forest"])
        pipe = Pipeline([("preproc", preproc), ("rf", rf)])
        pipe.fit(X_train, y_train)
        res_rf = evaluate_and_log(pipe, X_val, y_val, "random_forest", artifact_dir)

    # XGBoost
    with mlflow.start_run(run_name="xgboost"):
        xgb_params = cfg["training"]["xgboost"]["params"]
        xgbr = xgb.XGBRegressor(**xgb_params)
        pipe = Pipeline([("preproc", preproc), ("xgb", xgbr)])
        pipe.fit(X_train, y_train)
        res_xgb = evaluate_and_log(pipe, X_val, y_val, "xgboost", artifact_dir)

    # TabPFN (if available) - note: often memory-heavy
    res_tabpfn = None
    if HAS_TABPFN and cfg["training"].get("tabpfn", {}).get("use_tabpfn", False):
        with mlflow.start_run(run_name="tabpfn"):
            # TabPFN expects numpy arrays and may require normalization
            from tabpfn import TabPFNRegressor
            Xtr = X_train.values.astype(float)
            Xv = X_val.values.astype(float)
            model = TabPFNRegressor(N_ensembles=1)  # configuration simple
            model.fit(Xtr, y_train.values.astype(float))
            preds = model.predict(Xv)
            rmse = mean_squared_error(y_val, preds, squared=False)
            mae = mean_absolute_error(y_val, preds)
            r2 = r2_score(y_val, preds)
            mlflow.log_metric("rmse", float(rmse))
            mlflow.log_metric("mae", float(mae))
            mlflow.log_metric("r2", float(r2))
            # save via joblib
            p = os.path.join(artifact_dir, "tabpfn_model.joblib")
            joblib.dump(model, p)
            mlflow.log_artifact(p)
            res_tabpfn = {"rmse": rmse, "mae": mae, "r2": r2, "model_path": p}

    # Summary
    summary = {"rf": res_rf, "xgb": res_xgb, "tabpfn": res_tabpfn}
    pd.Series({
        "rf_rmse": res_rf["rmse"], "xgb_rmse": res_xgb["rmse"],
        "tabpfn_rmse": (res_tabpfn["rmse"] if res_tabpfn else None)
    }).to_csv(os.path.join(artifact_dir, "model_summary.csv"))
    print("Training completed. Results saved to", artifact_dir)
