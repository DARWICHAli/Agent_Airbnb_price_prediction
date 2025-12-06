# src/eda.py
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import numpy as np

def load_config(path="configs/base.yaml"):
    with open(path,"r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="paris")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    proc = os.path.join(cfg["data"]["processed_dir"],args.city)
    train = pd.read_parquet(os.path.join(proc, "train.parquet"))
    out_dir = os.path.join("artifacts","figures",args.city)
    os.makedirs(out_dir, exist_ok=True)

    # Price distribution
    plt.figure(figsize=(8,5))
    sns.histplot(train["price"], bins=80, log_scale=(False,True))
    plt.title("Price distribution (train)")
    plt.savefig(os.path.join(out_dir, "price_dist.png"), bbox_inches="tight")
    plt.close()

    # Boxplot by room_type
    if 'room_type' in train.columns:
        plt.figure(figsize=(8,5))
        sns.boxplot(x='room_type', y='price', data=train)
        plt.yscale('symlog')
        plt.title("Price by room type")
        plt.savefig(os.path.join(out_dir, "price_by_room_type.png"), bbox_inches="tight")
        plt.close()

    # Correlation heatmap for numeric features
    nums = train.select_dtypes(include=[np.number])
    corr = nums.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title("Feature correlation (train)")
    plt.savefig(os.path.join(out_dir, "corr_heatmap.png"), bbox_inches="tight")
    plt.close()

    # Save some stats
    stats = train.describe(include='all').transpose()
    stats.to_csv(os.path.join(out_dir, "stats.csv"))
    print("EDA outputs saved to artifacts/figures")
