"""Model curator — GBIF image fetch, background removal, optional TripoSR 3D generation."""

import os
import re
import shutil

import requests

# ---------------------------------------------------------------------------
# Optional TripoSR import (guarded)
# ---------------------------------------------------------------------------
TRIPOSR_AVAILABLE = False
try:
    from tsr.system import TSR as TSR_CLASS  # noqa: F401
    TRIPOSR_AVAILABLE = True
except ImportError:
    pass

_triposr_model = None  # lazy singleton


def _get_triposr_model():
    global _triposr_model
    if _triposr_model is None:
        from tsr.system import TSR as TSR_CLASS  # noqa: F811
        import torch
        _triposr_model = TSR_CLASS.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _triposr_model = _triposr_model.to(device)
        _triposr_model.renderer.set_chunk_size(8192)
    return _triposr_model


def _run_triposr(nobg_path: str, out_glb_path: str) -> bool:
    """Run TripoSR on a background-removed image and save a .glb file.

    Returns True on success, False on any error.
    """
    try:
        import torch
        from PIL import Image

        model = _get_triposr_model()
        device = next(model.parameters()).device

        image = Image.open(nobg_path).convert("RGBA")
        with torch.no_grad():
            scene_codes = model([image], device=device)
        meshes = model.extract_mesh(scene_codes, resolution=256)
        meshes[0].export(out_glb_path)
        return True
    except Exception as e:
        print(f"TripoSR error: {e}")
        return False


# ---------------------------------------------------------------------------
# GBIF image fetch
# ---------------------------------------------------------------------------
GBIF_MEDIA_URL = "https://api.gbif.org/v1/occurrence/search"


def _fetch_gbif_image_urls(taxon_key: str, n: int = 5) -> list[str]:
    """Return up to *n* image URLs from GBIF for the given taxon key."""
    urls: list[str] = []
    offset = 0
    limit = 20  # fetch more per request so we can skip non-image records
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
                if media.get("type") == "StillImage" and media.get("identifier"):
                    urls.append(media["identifier"])
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
    """Scan *curator_dir* and return a list of candidate dicts.

    Each dict:
      {taxon_key, candidate_index, has_original, has_nobg, has_model,
       original_url, nobg_url, model_url}
    The *_url fields contain relative paths suitable for the Flask route
    ``/curator/outputs/<taxon_key>/<filename>``.
    """
    candidates: list[dict] = []
    if not os.path.isdir(curator_dir):
        return candidates

    for taxon_key in sorted(os.listdir(curator_dir)):
        taxon_dir = os.path.join(curator_dir, taxon_key)
        if not os.path.isdir(taxon_dir):
            continue
        # Collect indices from existing original_ files
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
    triposr_available: bool = False,
) -> list[dict]:
    """Download images, remove backgrounds, optionally run TripoSR.

    Idempotent — skips files that already exist on disk.
    Returns list of candidate dicts (same schema as get_curator_status).
    """
    from rembg import remove as rembg_remove, new_session

    taxon_dir = os.path.join(output_dir, taxon_key)
    os.makedirs(taxon_dir, exist_ok=True)

    # Reuse a single rembg session for the batch
    rembg_session = new_session("u2net")

    # Fetch URLs (only if we don't already have enough originals)
    existing_originals = [
        f for f in os.listdir(taxon_dir) if f.startswith("original_")
    ]
    if len(existing_originals) < n_images:
        urls = _fetch_gbif_image_urls(taxon_key, n_images)
    else:
        # Reconstruct URLs list from filenames (we just need indices)
        urls = [None] * n_images  # placeholder — files already downloaded

    results: list[dict] = []
    for i in range(n_images):
        orig_path = os.path.join(taxon_dir, f"original_{i}.jpg")
        nobg_path = os.path.join(taxon_dir, f"nobg_{i}.png")
        model_path = os.path.join(taxon_dir, f"model_{i}.glb")

        # 1. Download original image
        if not os.path.exists(orig_path):
            if i < len(urls) and urls[i]:
                try:
                    img_resp = requests.get(urls[i], timeout=20)
                    img_resp.raise_for_status()
                    with open(orig_path, "wb") as f:
                        f.write(img_resp.content)
                except Exception as e:
                    print(f"  Download failed for {taxon_key}[{i}]: {e}")
                    continue
            else:
                continue  # No URL available

        # 2. Remove background
        if not os.path.exists(nobg_path) and os.path.exists(orig_path):
            try:
                with open(orig_path, "rb") as f:
                    img_bytes = f.read()
                result_bytes = rembg_remove(img_bytes, session=rembg_session)
                with open(nobg_path, "wb") as f:
                    f.write(result_bytes)
            except Exception as e:
                print(f"  rembg failed for {taxon_key}[{i}]: {e}")

        # 3. TripoSR (optional)
        if (
            triposr_available
            and TRIPOSR_AVAILABLE
            and not os.path.exists(model_path)
            and os.path.exists(nobg_path)
        ):
            _run_triposr(nobg_path, model_path)

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
    """Copy the chosen .glb into generator/models/ and patch config.py.

    Returns {status, pest_type, model_path} or {status, error}.
    """
    pest_type = species_info.get(taxon_key, {}).get("pest_type")
    if not pest_type:
        return {"status": "error", "error": f"Unknown taxon_key: {taxon_key}"}

    src_glb = os.path.join(curator_dir, taxon_key, f"model_{candidate_index}.glb")
    if not os.path.exists(src_glb):
        return {"status": "error", "error": f"Model file not found: {src_glb}"}

    dst_glb = os.path.join(models_dir, f"{pest_type}.glb")
    shutil.copy2(src_glb, dst_glb)

    # Relative path used in config.py (relative to project root)
    rel_path = os.path.join("generator", "models", f"{pest_type}.glb").replace("\\", "/")
    _update_config_pest_model_paths(config_path, pest_type, rel_path)

    return {"status": "ok", "pest_type": pest_type, "model_path": rel_path}


def _update_config_pest_model_paths(config_path: str, pest_type: str, glb_rel_path: str):
    """Regex-patch PEST_MODEL_PATHS in config.py for *pest_type*."""
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Match: "mouse": None,   or   "mouse": "some/path.glb",
    pattern = re.compile(
        r'("' + re.escape(pest_type) + r'"\s*:\s*)(?:None|"[^"]*")(.*)'
    )

    new_content, count = pattern.subn(
        lambda m: m.group(1) + f'"{glb_rel_path}"' + m.group(2),
        content,
    )

    if count == 0:
        print(f"WARNING: could not find '{pest_type}' entry in PEST_MODEL_PATHS")
        return

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(new_content)
