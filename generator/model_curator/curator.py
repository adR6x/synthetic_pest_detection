"""Model curator — GBIF image fetch, background removal, textured ellipsoid 3D generation.

Storage layout
--------------
outputs/curator/<taxon_key>/          <- gitignored; all temporary
    original_<i>.jpg                  <- downloaded image
    nobg_<i>.png                      <- background-removed image
    model_<i>.glb                     <- textured ellipsoid 3D model
    metadata.json                     <- maps index -> GBIF URL (temp)

generator/models/<pest_type>.glb      <- git-tracked; written only by keep_candidate()
generator/model_curator/discards.json <- git-tracked; GBIF URLs that were discarded
"""

import ast
import json
import os
import re
import shutil

import requests

# ---------------------------------------------------------------------------
# Ellipsoid GLB generation
# ---------------------------------------------------------------------------

# Pest-specific ellipsoid scales: (length_X, width_Y, height_Z)
_PEST_SCALES: dict[str, tuple[float, float, float]] = {
    "cockroach": (2.0, 1.0, 0.35),   # flat, wide oval
    "mouse":     (1.6, 0.8, 0.8),    # rounded, slightly elongated
    "rat":       (2.2, 0.9, 0.9),    # longer than mouse
}


def _generate_ellipsoid_glb(nobg_path: str, out_glb_path: str, pest_type: str = "") -> bool:
    """Project a background-removed image onto a UV ellipsoid and export as .glb."""
    try:
        import numpy as np
        import trimesh
        import trimesh.visual.material
        from PIL import Image

        sx, sy, sz = _PEST_SCALES.get(pest_type, (1.0, 1.0, 1.0))
        n_lat, n_lon = 32, 64

        verts, uvs = [], []
        for i in range(n_lat + 1):
            theta = np.pi * i / n_lat
            for j in range(n_lon + 1):
                phi = 2.0 * np.pi * j / n_lon
                verts.append([
                    np.sin(theta) * np.cos(phi) * sx,
                    np.sin(theta) * np.sin(phi) * sy,
                    np.cos(theta) * sz,
                ])
                uvs.append([j / n_lon, 1.0 - i / n_lat])

        faces = []
        for i in range(n_lat):
            for j in range(n_lon):
                a = i * (n_lon + 1) + j
                b, c, d = a + 1, a + (n_lon + 1), a + (n_lon + 1) + 1
                faces += [[a, c, b], [b, c, d]]

        verts = np.array(verts, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        uvs   = np.array(uvs,   dtype=np.float32)

        img      = Image.open(nobg_path).convert("RGBA")
        material = trimesh.visual.material.SimpleMaterial(image=img)
        visuals  = trimesh.visual.TextureVisuals(uv=uvs, material=material)
        mesh     = trimesh.Trimesh(vertices=verts, faces=faces,
                                   visual=visuals, process=False)
        mesh.export(out_glb_path)
        return True
    except Exception as e:
        print(f"Ellipsoid GLB error: {e}")
        return False


# ---------------------------------------------------------------------------
# Discard log helpers  (git-tracked: generator/model_curator/discards.json)
# ---------------------------------------------------------------------------

def _load_discards(discards_path: str) -> dict:
    """Return {taxon_key: [url, ...]} from discards.json, or {} if missing."""
    if os.path.exists(discards_path):
        with open(discards_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_discards(discards_path: str, discards: dict) -> None:
    with open(discards_path, "w", encoding="utf-8") as f:
        json.dump(discards, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Per-taxon metadata helpers  (temp: outputs/curator/<taxon>/metadata.json)
# Maps candidate index (str) -> original GBIF URL
# ---------------------------------------------------------------------------

def _load_metadata(taxon_dir: str) -> dict:
    path = os.path.join(taxon_dir, "metadata.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_metadata(taxon_dir: str, metadata: dict) -> None:
    path = os.path.join(taxon_dir, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# GBIF image fetch
# ---------------------------------------------------------------------------
GBIF_MEDIA_URL = "https://api.gbif.org/v1/occurrence/search"


def _fetch_gbif_image_urls(
    taxon_key: str,
    n: int = 5,
    exclude_urls: set | None = None,
) -> list[str]:
    """Return up to *n* image URLs from GBIF, skipping any in *exclude_urls*."""
    exclude_urls = exclude_urls or set()
    urls: list[str] = []
    offset = 0
    limit = 20
    while len(urls) < n:
        resp = requests.get(
            GBIF_MEDIA_URL,
            params={
                "taxonKey": taxon_key,
                "mediaType": "StillImage",
                "limit": limit,
                "offset": offset,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            break
        for occ in results:
            for media in occ.get("media", []):
                url = media.get("identifier")
                if media.get("type") == "StillImage" and url and url not in exclude_urls:
                    urls.append(url)
                    if len(urls) >= n:
                        return urls
        offset += limit
        if offset >= data.get("count", 0):
            break
    return urls


# ---------------------------------------------------------------------------
# Status scanner
# ---------------------------------------------------------------------------

def get_curator_status(curator_dir: str) -> list[dict]:
    """Scan outputs/curator/ and return candidate dicts for all present images.

    Discarded candidates are already deleted from disk, so they won't appear.
    """
    candidates: list[dict] = []
    if not os.path.isdir(curator_dir):
        return candidates

    for taxon_key in sorted(os.listdir(curator_dir)):
        taxon_dir = os.path.join(curator_dir, taxon_key)
        if not os.path.isdir(taxon_dir):
            continue
        indices: set[int] = set()
        for fname in os.listdir(taxon_dir):
            m = re.match(r"^original_(\d+)\.", fname)
            if m:
                indices.add(int(m.group(1)))
        for i in sorted(indices):
            orig_jpg = f"original_{i}.jpg"
            nobg_png = f"nobg_{i}.png"
            model_glb = f"model_{i}.glb"
            candidates.append({
                "taxon_key": taxon_key,
                "candidate_index": i,
                "has_original": os.path.exists(os.path.join(taxon_dir, orig_jpg)),
                "has_nobg": os.path.exists(os.path.join(taxon_dir, nobg_png)),
                "has_model": os.path.exists(os.path.join(taxon_dir, model_glb)),
                "original_url": orig_jpg,
                "nobg_url": nobg_png,
                "model_url": model_glb,
            })
    return candidates


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline_for_taxon(
    taxon_key: str,
    output_dir: str,
    n_images: int = 5,
    pest_type: str = "",
    discards_path: str | None = None,
) -> list[dict]:
    """Download images, remove backgrounds, generate textured ellipsoid GLB.

    - Skips GBIF URLs that are in discards.json.
    - Saves index→URL mapping to metadata.json so discards can be logged later.
    - Idempotent: skips files that already exist on disk.
    """
    from rembg import remove as rembg_remove, new_session

    taxon_dir = os.path.join(output_dir, taxon_key)
    os.makedirs(taxon_dir, exist_ok=True)

    # Load discarded URLs for this taxon
    discards = _load_discards(discards_path) if discards_path else {}
    discarded_urls: set[str] = set(discards.get(taxon_key, []))

    # Load existing index→URL metadata
    metadata = _load_metadata(taxon_dir)

    # Find existing downloaded indices
    existing_indices: set[int] = set()
    for fname in os.listdir(taxon_dir):
        m = re.match(r"^original_(\d+)\.", fname)
        if m:
            existing_indices.add(int(m.group(1)))

    need = n_images - len(existing_indices)
    next_index = max(existing_indices) + 1 if existing_indices else 0

    # Fetch new URLs only if we need more images
    new_urls: list[str] = []
    if need > 0:
        # Exclude both discarded URLs and URLs already on disk
        already_have: set[str] = set(metadata.values())
        exclude_urls = discarded_urls | already_have
        # Fetch extras to compensate for any that might be in the exclude set
        new_urls = _fetch_gbif_image_urls(
            taxon_key,
            need + len(exclude_urls),
            exclude_urls=exclude_urls,
        )
        new_urls = new_urls[:need]

    rembg_session = new_session("u2net")
    results: list[dict] = []

    # Process existing indices first (rembg / ellipsoid catch-up)
    all_indices = sorted(existing_indices) + list(range(next_index, next_index + len(new_urls)))

    for idx, i in enumerate(all_indices):
        orig_path = os.path.join(taxon_dir, f"original_{i}.jpg")
        nobg_path = os.path.join(taxon_dir, f"nobg_{i}.png")
        model_path = os.path.join(taxon_dir, f"model_{i}.glb")

        # Download if this is a new index
        if i >= next_index:
            url_idx = i - next_index
            if url_idx < len(new_urls):
                url = new_urls[url_idx]
                if not os.path.exists(orig_path):
                    try:
                        img_resp = requests.get(url, timeout=20)
                        img_resp.raise_for_status()
                        with open(orig_path, "wb") as f:
                            f.write(img_resp.content)
                        # Save URL to metadata for future discard logging
                        metadata[str(i)] = url
                        _save_metadata(taxon_dir, metadata)
                    except Exception as e:
                        print(f"  Download failed for {taxon_key}[{i}]: {e}")
                        continue

        # Background removal
        if not os.path.exists(nobg_path) and os.path.exists(orig_path):
            try:
                with open(orig_path, "rb") as f:
                    img_bytes = f.read()
                result_bytes = rembg_remove(img_bytes, session=rembg_session)
                with open(nobg_path, "wb") as f:
                    f.write(result_bytes)
            except Exception as e:
                print(f"  rembg failed for {taxon_key}[{i}]: {e}")

        # Ellipsoid 3D model (always runs when nobg exists)
        if not os.path.exists(model_path) and os.path.exists(nobg_path):
            _generate_ellipsoid_glb(nobg_path, model_path, pest_type)

        results.append({
            "taxon_key": taxon_key,
            "candidate_index": i,
            "has_original": os.path.exists(orig_path),
            "has_nobg": os.path.exists(nobg_path),
            "has_model": os.path.exists(model_path),
            "original_url": f"original_{i}.jpg",
            "nobg_url": f"nobg_{i}.png",
            "model_url": f"model_{i}.glb",
        })

    return results


# ---------------------------------------------------------------------------
# Discard candidate
# ---------------------------------------------------------------------------

def discard_candidate(
    taxon_key: str,
    candidate_index: int,
    curator_dir: str,
    discards_path: str,
) -> dict:
    """Delete candidate files and log its GBIF URL to discards.json.

    - Deletes original_<i>.jpg, nobg_<i>.png, model_<i>.glb from outputs/curator/
    - Logs the GBIF URL (from metadata.json) to discards.json so it is never
      re-downloaded in future pipeline runs.
    - discards.json is git-tracked; outputs/curator/ is gitignored.
    """
    taxon_dir = os.path.join(curator_dir, taxon_key)
    metadata = _load_metadata(taxon_dir)
    url = metadata.get(str(candidate_index))

    # Log URL to discards.json
    if url:
        discards = _load_discards(discards_path)
        bucket = discards.setdefault(taxon_key, [])
        if url not in bucket:
            bucket.append(url)
        _save_discards(discards_path, discards)

    # Delete candidate files from temp storage
    for filename in [
        f"original_{candidate_index}.jpg",
        f"nobg_{candidate_index}.png",
        f"model_{candidate_index}.glb",
    ]:
        fpath = os.path.join(taxon_dir, filename)
        if os.path.exists(fpath):
            os.remove(fpath)

    # Remove index from metadata
    if str(candidate_index) in metadata:
        del metadata[str(candidate_index)]
        _save_metadata(taxon_dir, metadata)

    return {"status": "ok", "url_logged": url}


# ---------------------------------------------------------------------------
# Keep candidate
# ---------------------------------------------------------------------------

def keep_candidate(
    taxon_key: str,
    candidate_index: int,
    curator_dir: str,
    models_dir: str,
    config_path: str,
    species_info: dict,
) -> dict:
    """Copy the chosen .glb into generator/models/<pest_type>/ and append to config.py.

    Each kept model gets a unique filename (<taxon_key>_<candidate_index>.glb)
    so multiple keeps never overwrite each other.
    """
    pest_type = species_info.get(taxon_key, {}).get("pest_type")
    if not pest_type:
        return {"status": "error", "error": f"Unknown taxon_key: {taxon_key}"}

    src_glb = os.path.join(curator_dir, taxon_key, f"model_{candidate_index}.glb")
    if not os.path.exists(src_glb):
        return {"status": "error", "error": f"Model file not found: {src_glb}"}

    # Save into per-pest subdirectory with a unique filename
    pest_dir = os.path.join(models_dir, pest_type)
    os.makedirs(pest_dir, exist_ok=True)
    filename = f"{taxon_key}_{candidate_index}.glb"
    dst_glb = os.path.join(pest_dir, filename)
    shutil.copy2(src_glb, dst_glb)

    rel_path = f"generator/models/{pest_type}/{filename}"
    _append_config_pest_model_path(config_path, pest_type, rel_path)

    return {"status": "ok", "pest_type": pest_type, "model_path": rel_path}


def _append_config_pest_model_path(config_path: str, pest_type: str, new_glb_path: str):
    """Append *new_glb_path* to the PEST_MODEL_PATHS list for *pest_type* in config.py."""
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Match:  "mouse": [],   or   "mouse": ["path1", "path2"],
    pattern = re.compile(
        r'("' + re.escape(pest_type) + r'"\s*:\s*)(\[[^\[\]]*\])(.*)'
    )

    def replacer(m):
        current_list = ast.literal_eval(m.group(2))
        if new_glb_path not in current_list:
            current_list.append(new_glb_path)
        return m.group(1) + json.dumps(current_list) + m.group(3)

    new_content, count = pattern.subn(replacer, content)

    if count == 0:
        print(f"WARNING: could not find '{pest_type}' list in PEST_MODEL_PATHS")
        return

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(new_content)
