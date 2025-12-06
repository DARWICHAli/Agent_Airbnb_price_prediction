# src/agent.py
import argparse, subprocess, os
import yaml
from jinja2 import Template

def run_stage(stage, city):
    env = os.environ.copy()
    # call sequentially
    if stage in ("all","fetch"):
        subprocess.check_call(["python","-u","src/fetcher.py","--city",city])
    if stage in ("all","preprocess"):
        subprocess.check_call(["python","-u","src/preprocess.py","--city",city])
    if stage in ("all","eda"):
        subprocess.check_call(["python","-u","src/eda.py","--city",city])
    if stage in ("all","train"):
        subprocess.check_call(["python","-u","src/train.py","--city",city,"--config","configs/base.yaml"])
    if stage in ("all","eval"):
        subprocess.check_call(["python","-u","src/evaluate.py","--city",city])
    if stage in ("all","report","generate_report"):
        # build a markdown report from templates + artifacts
        template_md = """
# Airbnb Price Prediction Report - {{city}}

## Executive summary
- City: {{city}}

## Data
- Source: InsideAirbnb (latest available snapshot). See: https://insideairbnb.com/get-the-data/

## EDA
![price_dist](artifacts/figures/price_dist.png)

## Models & Results
- Holdout results: see `artifacts/holdout_results.csv`

## Recommendations
- (auto-generated)
"""
        content = Template(template_md).render(city=city)
        os.makedirs("reports", exist_ok=True)
        md_path = os.path.join("reports", f"{city}_report.md")
        with open(md_path,"w") as f:
            f.write(content)
        # Convert to PDF if pandoc available
        pdf_path = os.path.join("reports", f"{city}_report.pdf")
        try:
            subprocess.check_call(["pandoc", md_path, "-o", pdf_path])
            print("PDF created:", pdf_path)
        except Exception:
            print("Pandoc not available—report left as markdown:", md_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="paris")
    parser.add_argument("--stage", default="all", choices=["all","fetch","preprocess","eda","train","eval","report","generate_report"])
    args = parser.parse_args()
    run_stage(args.stage, args.city)
