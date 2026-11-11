"""Download the afhubbard/gpu-prices parquet snapshots into ./data (skips files already present)."""
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "afhubbard/gpu-prices"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

meta = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}"))
files = [s["rfilename"] for s in meta["siblings"] if s["rfilename"].endswith(".parquet")]


def fetch(rel):
    dest = os.path.join(OUT, rel)
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{rel}"
    urllib.request.urlretrieve(url, dest + ".tmp")
    os.replace(dest + ".tmp", dest)


with ThreadPoolExecutor(8) as ex:
    list(ex.map(fetch, files))
print(f"{len(files)} files in {OUT}")
