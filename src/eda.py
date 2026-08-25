# src/eda.py
import argparse, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

sns.set_theme(style="whitegrid", palette="muted")

def load_config(path="configs/base.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def city_title(city):
    return city.replace("-", " ").title()


def load_full_data(proc_dir):
    """Concatenate the train/val/test splits to get the full cleaned dataset."""
    parts = []
    for split in ("train", "val", "test"):
        p = os.path.join(proc_dir, f"{split}.parquet")
        if os.path.exists(p):
            parts.append(pd.read_parquet(p))
    if not parts:
        raise SystemExit(f"No processed parquet files found in {proc_dir}. Run preprocess first.")
    return pd.concat(parts, ignore_index=True)


def price_dist_plot(df, out_dir, city):
    price = df["price"]
    median, mean = price.median(), price.mean()
    p25, p75 = price.quantile(0.25), price.quantile(0.75)

    plt.figure(figsize=(9, 5))
    sns.histplot(price, bins=80, color="#4C72B0", edgecolor="white", linewidth=0.3)
    plt.axvline(median, color="#C44E52", ls="-", lw=1.6, label=f"Median: ${median:,.0f}")
    plt.axvline(mean, color="#DD8452", ls="--", lw=1.6, label=f"Mean: ${mean:,.0f}")
    plt.axvline(p25, color="gray", ls=":", lw=1.2, label=f"P25: ${p25:,.0f}")
    plt.axvline(p75, color="gray", ls=":", lw=1.2, label=f"P75: ${p75:,.0f}")
    plt.xlabel("Nightly price ($)")
    plt.ylabel("Number of listings")
    plt.title(f"Nightly price distribution — {city_title(city)} ({len(df):,} listings)")
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "price_dist.png"), bbox_inches="tight", dpi=150)
    plt.close()


def price_by_room_type_plot(df, out_dir, city):
    if "room_type" not in df.columns:
        return
    # order categories by median price, show flier-free boxes + mean markers
    medians = df.groupby("room_type")["price"].median().sort_values()
    order = medians.index.tolist()
    counts = df["room_type"].value_counts()

    plt.figure(figsize=(9, 5))
    sns.boxplot(x="room_type", y="price", data=df, order=order, showfliers=False,
                color="#8DA0CB", width=0.55)
    sns.pointplot(x="room_type", y="price", data=df, order=order, estimator=np.mean,
                  color="#C44E52", markers="D", errorbar=None)
    q3 = df.groupby("room_type")["price"].quantile(0.75)
    cap = float(df["price"].quantile(0.98))
    capped = bool(q3.max() > cap)
    y_top = cap if capped else q3.max() * 1.18
    plt.ylim(top=y_top)
    for i, cat in enumerate(order):
        label_y = min(q3[cat] * 1.06, y_top * 0.97)
        plt.text(i, label_y, f"${medians[cat]:,.0f}", ha="center",
                 fontsize=9, color="#333333", fontweight="bold")
    if capped:
        plt.text(0.99, 0.985,
                 "Note: axis capped at the 98th percentile of price for readability;\nsome room types extend higher",
                 transform=plt.gca().transAxes, ha="right", va="top",
                 fontsize=8, color="#666666")
    labels = [f"{c}\n(n={counts[c]:,})" for c in order]
    plt.xticks(range(len(order)), labels)
    plt.xlabel("Room type")
    plt.ylabel("Nightly price ($)")
    plt.title(f"Nightly price by room type — {city_title(city)}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "price_by_room_type.png"), bbox_inches="tight", dpi=150)
    plt.close()


def corr_heatmap(df, out_dir, city):
    nums = df.select_dtypes(include=[np.number])
    if nums.shape[1] < 2:
        return
    corr = nums.corr()
    k = corr.shape[0]
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # show lower triangle only
    plt.figure(figsize=(max(9, k * 0.55), max(7, k * 0.5)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, mask=mask,
                square=True, linewidths=0.4, annot_kws={"size": 7})
    plt.title(f"Correlation of numeric features — {city_title(city)}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "corr_heatmap.png"), bbox_inches="tight", dpi=150)
    plt.close()


def price_by_bedrooms_plot(df, out_dir, city):
    if "bedrooms" not in df.columns:
        return
    clipped = df["bedrooms"].clip(upper=8)  # group 8+ into a single "8+" bin
    g = df.groupby(clipped)["price"].agg(["median", "count"])

    labels = [f"{int(i)}" if i < 8 else "8+" for i in g.index]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(range(len(g)), g["median"], color="#4C72B0", width=0.7)
    for bar, med, cnt in zip(bars, g["median"], g["count"]):
        plt.text(bar.get_x() + bar.get_width() / 2, med * 1.03, f"${med:,.0f}",
                 ha="center", fontsize=9, color="#333333", fontweight="bold")
        plt.text(bar.get_x() + bar.get_width() / 2, 0, f"n={cnt:,}", ha="center",
                 fontsize=7.5, color="white", va="bottom")
    plt.xticks(range(len(g)), labels)
    plt.xlabel("Bedrooms")
    plt.ylabel("Median nightly price ($)")
    plt.title(f"Median nightly price by number of bedrooms — {city_title(city)}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "price_by_bedrooms.png"), bbox_inches="tight", dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="paris")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    proc = os.path.join(cfg["data"]["processed_dir"], args.city)
    out_dir = os.path.join("artifacts", "figures", args.city)
    os.makedirs(out_dir, exist_ok=True)

    df = load_full_data(proc)
    print(f"EDA on full cleaned dataset: {len(df):,} rows, {df.shape[1]} columns")

    price_dist_plot(df, out_dir, args.city)
    price_by_room_type_plot(df, out_dir, args.city)
    corr_heatmap(df, out_dir, args.city)
    price_by_bedrooms_plot(df, out_dir, args.city)

    # Summary stats (full dataset) for the report
    stats = df.describe(include="all").transpose()
    stats.to_csv(os.path.join(out_dir, "stats.csv"))

    # Per-room-type stats for the report table
    if "room_type" in df.columns:
        rt = df.groupby("room_type")["price"].agg(["count", "mean", "median", "min", "max"])
        rt = rt.round(2).sort_values("median", ascending=False)
        rt.to_csv(os.path.join(out_dir, "room_type_stats.csv"))

    print("EDA outputs saved to", out_dir)
