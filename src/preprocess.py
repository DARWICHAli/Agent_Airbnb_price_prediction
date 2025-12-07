# src/preprocess.py
import argparse, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

def load_config(path="configs/base.yaml"):
    with open(path,"r") as f:
        return yaml.safe_load(f)

def basic_clean(df, cfg):
    # standard price column: remove currency symbols and commas
    price_col = cfg["preprocessing"]["price_col"]
    df[price_col] = df[price_col].astype(str).str.replace(r"[$,]", "", regex=True)
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    # keep only required columns present
    req = [c for c in cfg["data"]["required_columns"] if c in df.columns]
    df = df[req].copy()
    today = pd.Timestamp.today()
    # convert numeric columns
    for c in df.select_dtypes(include=["object"]).columns:
        if c in ["latitude", "longitude"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if c == "host_since":
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df["host_months_since"] = (today.year - df[c].dt.year) * 12 + (today.month - df[c].dt.month)
            df[c] = df[c].astype('int64')
        if c in ["host_response_rate", "host_acceptance_rate"]:
            df[c] = df[c].str.rstrip('%')
            df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
        if c in ["host_is_superhost","instant_bookable",""]:
            df[c] = df[c].map({'t':1, 'f':0})
        if c in ["host_response_time","host_neighbourhood","property_type","room_type"]:
            df[c] = df[c].astype(str)

    # basic imputation
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    num_fill = df[num_cols].median()
    df[num_cols] = df[num_cols].fillna(num_fill)
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "missing")
    return df

def make_splits(df, cfg):
    s = cfg["split"]
    train_frac = s["train"]
    val_frac = s["val"]
    test_frac = s["test"]
    X = df.drop(columns=[cfg["training"]["target"]])
    y = df[cfg["training"]["target"]]
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=train_frac, random_state=cfg["seed"])
    # split remainder proportionally
    val_rel = val_frac / (val_frac + test_frac)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size=val_rel, random_state=cfg["seed"])
    return X_train.join(y_train), X_val.join(y_val), X_test.join(y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="paris")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    raw_dir = cfg["data"]["raw_dir"]
    processed_dir = os.path.join(cfg["data"]["processed_dir"],args.city)
    os.makedirs(processed_dir, exist_ok=True)
    # find latest listings file for city in raw_dir
    candidate = None
    for f in os.listdir(raw_dir):
        if "listings" in f and args.city.lower() in f.lower():
            candidate = os.path.join(raw_dir, f)
    if not candidate:
        raise SystemExit("No listings file found in raw dir. Run fetcher first.")
    df = pd.read_csv(candidate)
    df_clean = basic_clean(df, cfg)
    train, val, test = make_splits(df_clean, cfg)
    train.to_parquet(os.path.join(processed_dir, "train.parquet"))
    val.to_parquet(os.path.join(processed_dir, "val.parquet"))
    test.to_parquet(os.path.join(processed_dir, "test.parquet"))
    print("Preprocessing done. Files saved in", processed_dir)
