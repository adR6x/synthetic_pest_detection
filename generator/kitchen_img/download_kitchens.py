"""Download 100 kitchen images from Places365 standard validation set.

Strategy:
  1. Stream filelist tar → extract places365_val.txt in memory
  2. Filter for category 203 (kitchen) to get target filenames
  3. Stream val_256.tar → save only the kitchen images, stop at 100

No API key required. All data is MIT/CSAIL public.
"""

import io
import os
import random
import tarfile
import urllib.request

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET = 100
KITCHEN_CATEGORY = 203   # /k/kitchen in Places365-standard

FILELIST_URL = "https://data.csail.mit.edu/places/places365/filelist_places365-standard.tar"
VAL_IMAGES_URL = "https://data.csail.mit.edu/places/places365/val_256.tar"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-downloader/1.0)"}


def fetch_bytes(url, label):
    print(f"  Downloading {label} ...", flush=True)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def get_kitchen_filenames():
    """Return set of Places365_val_XXXXXXXX.jpg filenames that are category 203."""
    data = fetch_bytes(FILELIST_URL, "filelist tar (67 MB)")
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        val_member = t.getmember("places365_val.txt")
        val_txt = t.extractfile(val_member).read().decode("utf-8")

    kitchen = set()
    for line in val_txt.strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 2 and int(parts[1]) == KITCHEN_CATEGORY:
            kitchen.add(parts[0])   # e.g. "Places365_val_00012345.jpg"

    print(f"  {len(kitchen)} kitchen images in validation set.")
    return kitchen


def stream_download_kitchen(kitchen_files):
    """Stream the val_256.tar and save only the kitchen images."""
    print(f"  Streaming val_256.tar (525 MB) — saving kitchen images ...", flush=True)
    req = urllib.request.Request(VAL_IMAGES_URL, headers=HEADERS)

    saved = 0
    seen = 0
    with urllib.request.urlopen(req, timeout=300) as r:
        with tarfile.open(fileobj=r, mode="r|") as tar:
            for member in tar:
                seen += 1
                name = os.path.basename(member.name)
                if not member.isfile() or name not in kitchen_files:
                    continue

                fobj = tar.extractfile(member)
                if fobj is None:
                    continue

                out_path = os.path.join(OUT_DIR, f"kitchen_{saved+1:03d}_{name}")
                with open(out_path, "wb") as f:
                    f.write(fobj.read())
                saved += 1
                print(f"  [{saved}/{len(kitchen_files)}] {name}", flush=True)

                if saved >= len(kitchen_files):
                    break

    print(f"\nDone — {saved} images saved to:\n  {OUT_DIR}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    kitchen_files = get_kitchen_filenames()
    stream_download_kitchen(kitchen_files)


if __name__ == "__main__":
    main()
