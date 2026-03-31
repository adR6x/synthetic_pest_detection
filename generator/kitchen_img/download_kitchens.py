"""Download kitchen images from Places365 standard validation set.

Strategy:
  1. Stream filelist tar → extract places365_val.txt in memory
  2. Filter for category 203 (kitchen) to get target filenames
  3. Skip filenames we have seen before (downloaded or curator-deleted)
  4. Stream val_256.tar → save only unseen kitchen images, stop at TARGET

No API key required. All data is MIT/CSAIL public.
"""

import io
import json
import os
import random
import re
import tarfile
import urllib.request

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET = 100
RANDOM_SEED = 42
KITCHEN_CATEGORY = 203   # /k/kitchen in Places365-standard
STATE_FILE = os.path.join(OUT_DIR, "download_state.json")

FILELIST_URL = "https://data.csail.mit.edu/places/places365/filelist_places365-standard.tar"
VAL_IMAGES_URL = "https://data.csail.mit.edu/places/places365/val_256.tar"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-downloader/1.0)"}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PLACES_NAME_RE = re.compile(r"(Places365_val_\d{8}\.jpg)$")
OUTPUT_INDEX_RE = re.compile(r"^kitchen_(\d+)_")


def fetch_bytes(url, label):
    print(f"  Downloading {label} ...", flush=True)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def extract_places_filename(name):
    """Extract canonical Places filename (e.g. Places365_val_00012345.jpg)."""
    m = PLACES_NAME_RE.search(name)
    return m.group(1) if m else None


def list_local_kitchen_images():
    """Return local kitchen image files in OUT_DIR."""
    files = []
    for name in sorted(os.listdir(OUT_DIR)):
        path = os.path.join(OUT_DIR, name)
        if os.path.isfile(path) and name.lower().endswith(IMAGE_EXTS):
            files.append(name)
    return files


def _load_seen_from_state():
    if not os.path.exists(STATE_FILE):
        return set()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return set()
    values = data.get("seen_places365_files", [])
    if not isinstance(values, list):
        return set()
    return {v for v in values if isinstance(v, str)}


def _save_seen_to_state(seen_places):
    payload = {"seen_places365_files": sorted(seen_places)}
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _seen_from_existing_files():
    seen = set()
    for name in list_local_kitchen_images():
        places_name = extract_places_filename(name)
        if places_name:
            seen.add(places_name)
    return seen


def get_seen_places():
    """Union of state file + current images on disk."""
    return _load_seen_from_state() | _seen_from_existing_files()


def mark_places_as_seen(places_filenames):
    """Persist given Places filenames so they are never re-downloaded."""
    clean = {p for p in places_filenames if isinstance(p, str) and p}
    if not clean:
        return
    seen = get_seen_places()
    seen.update(clean)
    _save_seen_to_state(seen)


def next_output_index():
    """Return next sequential output index for kitchen_<idx>_*.jpg filenames."""
    max_idx = 0
    for name in os.listdir(OUT_DIR):
        m = OUTPUT_INDEX_RE.match(name)
        if not m:
            continue
        max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


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


def stream_download_kitchen(kitchen_files, start_index):
    """Stream the val_256.tar and save only the kitchen images."""
    print(
        f"  Streaming val_256.tar (525 MB) — saving up to {len(kitchen_files)} images ...",
        flush=True,
    )
    req = urllib.request.Request(VAL_IMAGES_URL, headers=HEADERS)

    saved = 0
    saved_places = set()
    out_index = start_index
    with urllib.request.urlopen(req, timeout=300) as r:
        with tarfile.open(fileobj=r, mode="r|") as tar:
            for member in tar:
                name = os.path.basename(member.name)
                if not member.isfile() or name not in kitchen_files:
                    continue

                fobj = tar.extractfile(member)
                if fobj is None:
                    continue

                out_path = os.path.join(OUT_DIR, f"kitchen_{out_index:03d}_{name}")
                with open(out_path, "wb") as f:
                    f.write(fobj.read())
                saved += 1
                out_index += 1
                saved_places.add(name)
                print(f"  [{saved}/{len(kitchen_files)}] {name}", flush=True)

                if saved >= len(kitchen_files):
                    break

    print(f"\nDone — {saved} images saved to:\n  {OUT_DIR}")
    return saved_places


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    kitchen_files = get_kitchen_filenames()
    seen_places = get_seen_places()
    unseen_kitchen_files = sorted(kitchen_files - seen_places)
    print(f"  Already seen (downloaded or curated): {len(seen_places)}")

    if not unseen_kitchen_files:
        print("  No unseen kitchen images left to download.")
        return

    target_count = min(TARGET, len(unseen_kitchen_files))
    rng = random.Random(RANDOM_SEED)
    selected_files = set(rng.sample(unseen_kitchen_files, target_count))
    print(f"  Selecting {target_count} unseen images (seed={RANDOM_SEED}).")

    start_index = next_output_index()
    saved_places = stream_download_kitchen(selected_files, start_index=start_index)
    mark_places_as_seen(saved_places)


if __name__ == "__main__":
    main()
