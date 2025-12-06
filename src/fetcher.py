# src/fetcher.py
import argparse, os, gzip, shutil, re
import requests
from bs4 import BeautifulSoup

def find_city_listing_link(html, city):
    soup = BeautifulSoup(html, "html.parser")
    # La page regroupe des lignes par ville; on cherche span/text contenant city (cas-insensible)
    anchors = soup.find_all("a", href=True)
    city_lower = city.lower()
    # heuristique: les fichiers contiennent 'listings.csv.gz' et commencent par Cityname
    for a in anchors:
        href = a['href']
        text = (a.get_text() or "").lower()
        if "listings.csv" in href and city_lower in href.lower():
            return href
        if "listings.csv" in href and city_lower in text:
            return href
    return None

def download(url, out_path):
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def main(city, base_url, raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    r = requests.get(base_url, timeout=30)
    r.raise_for_status()
    link = find_city_listing_link(r.text, city)
    if not link:
        # second attempt: find link that contains city without exact match
        raise SystemExit(f"Listings link for city '{city}' not found on {base_url}")
    # some links are relative
    if link.startswith('/'):
        base = "https://insideairbnb.com"
        link = base + link

    filename = city + "_" + os.path.basename(link)
    out_file = os.path.join(raw_dir, filename)
    print(f"Downloading {link} -> {out_file}")
    download(link, out_file)
    # if gz, extract
    if out_file.endswith(".gz"):
        dest = out_file[:-3]
        with gzip.open(out_file, "rb") as f_in, open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"Extracted to {dest}")
    else:
        dest = out_file
    print("Fetch complete:", dest)
    return dest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="paris")
    parser.add_argument("--base_url", default="https://insideairbnb.com/get-the-data/")
    parser.add_argument("--raw_dir", default="data/raw")
    args = parser.parse_args()
    main(args.city, args.base_url, args.raw_dir)
