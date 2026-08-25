# src/train.py
import argparse, os, yaml
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import mlflow
from mlflow.tracking import MlflowClient
import joblib

# TabPFN is optional (heavyweight transformer-based baseline)
try:
    from tabpfn import TabPFNRegressor
    HAS_TABPFN = True
except Exception as e:
    HAS_TABPFN = False
    print("TabPFN not available, will skip:", e)

RF_PARAM_GRID = {
    "rf__n_estimators": [100, 200, 400],
    "rf__max_depth": [None, 10, 20],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__max_features": ["sqrt"],
}

XGB_PARAM_GRID = {
    "xgb__n_estimators": [300, 500, 700],
    "xgb__max_depth": [4, 6, 8],
    "xgb__learning_rate": [0.03, 0.05, 0.1],
    "xgb__subsample": [0.8, 1.0],
    "xgb__colsample_bytree": [0.8, 1.0],
}


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


def kill_stale_runs(client, experiment_id):
    """Mark RUNNING runs (left by crashed sessions) as KILLED."""
    try:
        for run in client.search_runs(experiment_id, filter_string="status = 'RUNNING'"):
            client.set_terminated(run.info.run_id, status="KILLED")
            print("Killed stale run:", run.info.run_id)
    except Exception as e:
        print("Could not clean stale runs:", e)


def tune_pipeline(estimator, param_grid, preproc, X_train, y_train, cfg, model_name):
    """Randomized search on a subsample, then refit the best pipeline on full train."""
    tun = cfg["training"].get("tuning", {})
    pipe = Pipeline([("preproc", preproc), (model_name, estimator)])
    if not tun.get("enabled", True):
        pipe.fit(X_train, y_train)
        return pipe, {}

    rng = np.random.RandomState(tun.get("random_state", cfg["seed"]))
    n_max = tun.get("max_samples", 10000)
    X_s, y_s = X_train, y_train
    if len(X_s) > n_max:
        idx = rng.choice(len(X_s), n_max, replace=False)
        X_s, y_s = X_s.iloc[idx], y_s.iloc[idx]

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_grid,
        n_iter=tun.get("n_iter", 4),
        cv=tun.get("cv", 3),
        scoring=tun.get("scoring", "neg_root_mean_squared_error"),
        n_jobs=-1,
        random_state=tun.get("random_state", cfg["seed"]),
        refit=True,
    )
    search.fit(X_s, y_s)
    best_pipe = search.best_estimator_
    best_pipe.fit(X_train, y_train)  # refit on the full training set
    print(f"Best {model_name} params: {search.best_params_}")
    return best_pipe, search.best_params_


def evaluate_and_log(model, X_val, y_val, run_name, artifact_dir, city, extra_params=None):
    preds = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, preds)
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    mlflow.set_tag("city", city)
    mlflow.log_metric("rmse", float(rmse))
    mlflow.log_metric("mae", float(mae))
    mlflow.log_metric("r2", float(r2))
    if extra_params:
        mlflow.log_params(extra_params)

    # save model
    model_path = os.path.join(artifact_dir, f"{run_name}.joblib")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    return {"rmse": rmse, "mae": mae, "r2": r2, "model_path": model_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="paris", type=str, required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    city = args.city
    cfg = load_config(args.config)

    proc = os.path.join(cfg["data"]["processed_dir"], city)
    train = pd.read_parquet(os.path.join(proc, "train.parquet"))
    val = pd.read_parquet(os.path.join(proc, "val.parquet"))
    target = cfg["training"]["target"]

    X_train, y_train, num_cols, cat_cols = get_feature_sets(train, target)
    X_val, y_val, _, _ = get_feature_sets(val, target)

    preproc = build_preprocessor(num_cols, cat_cols)

    artifact_dir = os.path.join(cfg["output"]["artifacts_dir"], city)
    os.makedirs(artifact_dir, exist_ok=True)

    mlflow.set_tracking_uri(cfg["experiment"]["mlflow_tracking_uri"])
    mlflow.set_experiment(cfg["experiment"]["experiment_name"])
    client = MlflowClient()
    exp = client.get_experiment_by_name(cfg["experiment"]["experiment_name"])
    if exp:
        kill_stale_runs(client, exp.experiment_id)

    # ===============================
    # Random Forest
    # ===============================
    with mlflow.start_run(run_name="random_forest"):
        rf_defaults = dict(cfg["training"]["random_forest"])
        rf_defaults.setdefault("n_jobs", -1)
        rf = RandomForestRegressor(**rf_defaults)
        pipe_rf, rf_params = tune_pipeline(rf, RF_PARAM_GRID, preproc, X_train, y_train, cfg, "rf")
        res_rf = evaluate_and_log(pipe_rf, X_val, y_val, "random_forest", artifact_dir, city, rf_params)

    # ===============================
    # XGBoost
    # ===============================
    with mlflow.start_run(run_name="xgboost"):
        xgb_params = dict(cfg["training"]["xgboost"]["params"])
        xgb_params.setdefault("n_jobs", -1)
        xgbr = xgb.XGBRegressor(**xgb_params)
        pipe_xgb, best_xgb = tune_pipeline(xgbr, XGB_PARAM_GRID, preproc, X_train, y_train, cfg, "xgb")
        res_xgb = evaluate_and_log(pipe_xgb, X_val, y_val, "xgboost", artifact_dir, city, best_xgb)

    # ===============================
    # TabPFN (optional, CPU-heavy: uses a capped training context)
    # ===============================
    res_tabpfn = None
    if HAS_TABPFN and cfg["training"].get("tabpfn", {}).get("use_tabpfn", False):
        with mlflow.start_run(run_name="tabpfn"):
            mlflow.set_tag("city", city)
            tp_cfg = cfg["training"]["tabpfn"]
            rng = np.random.RandomState(cfg["seed"])
            max_train = tp_cfg.get("max_train_samples", 1024)
            max_eval = tp_cfg.get("max_eval_samples", 1024)
            n_est = tp_cfg.get("n_estimators", 2)

            Xtr, ytr = X_train.copy(), y_train.copy()
            if len(Xtr) > max_train:
                idx = rng.choice(len(Xtr), max_train, replace=False)
                Xtr, ytr = Xtr.iloc[idx], ytr.iloc[idx]
            Xv, yv = X_val.copy(), y_val.copy()
            if len(Xv) > max_eval:
                idx = rng.choice(len(Xv), max_eval, replace=False)
                Xv, yv = Xv.iloc[idx], yv.iloc[idx]

            # TabPFN expects plain numpy matrices; fit its own preprocessor on the subsample
            Xtr_t = preproc.fit_transform(Xtr).astype(float)
            Xv_t = preproc.transform(Xv).astype(float)

            model = TabPFNRegressor(
                n_estimators=n_est,
                device="cpu",
                random_state=cfg["seed"],
                ignore_pretraining_limits=True,
            )
            model.fit(Xtr_t, ytr.values.astype(float))
            preds = model.predict(Xv_t)

            rmse = root_mean_squared_error(yv, preds)
            mae = mean_absolute_error(yv, preds)
            r2 = r2_score(yv, preds)

            mlflow.log_metric("rmse", float(rmse))
            mlflow.log_metric("mae", float(mae))
            mlflow.log_metric("r2", float(r2))
            mlflow.log_params({
                "n_train_samples": len(Xtr),
                "n_eval_samples": len(Xv),
                "n_estimators": n_est,
            })

            p = os.path.join(artifact_dir, "tabpfn_model.joblib")
            joblib.dump(model, p)
            mlflow.log_artifact(p)
            res_tabpfn = {"rmse": rmse, "mae": mae, "r2": r2, "model_path": p}
            print(f"TabPFN done on {len(Xtr)} train / {len(Xv)} eval samples")

    # Summary
    summary = pd.Series({
        "rf_rmse": res_rf["rmse"],
        "xgb_rmse": res_xgb["rmse"],
        "tabpfn_rmse": (res_tabpfn["rmse"] if res_tabpfn else None),
    })
    summary.to_csv(os.path.join(artifact_dir, "model_summary.csv"))
    print("Training completed. Results saved to", artifact_dir)
    print(summary.to_string())
