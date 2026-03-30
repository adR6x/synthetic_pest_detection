"""Flask web application — upload kitchen images and generate synthetic pest videos."""

import json
import os
import shutil
import tempfile
import queue
import threading
import time
import traceback

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify

from generator.config import UPLOAD_DIR, FRAMES_DIR, VIDEOS_DIR, LABELS_DIR, OUTPUT_DIR, PROJECT_ROOT
from generator.depth_estimator import preload_models
from generator.pipeline import generate_video
from generator.model_curator.species_info import SPECIES_INFO
from generator.model_curator.curator import (
    get_curator_status,
    run_pipeline_for_taxon,
    keep_candidate,
    discard_candidate,
)

# --- Curator constants ---
CURATOR_DIR = os.path.join(PROJECT_ROOT, "outputs", "curator")
MODELS_DIR = os.path.join(PROJECT_ROOT, "generator", "models")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "generator", "config.py")
# Git-tracked: logs discarded GBIF URLs so they are never re-downloaded
DISCARDS_PATH = os.path.join(PROJECT_ROOT, "generator", "model_curator", "discards.json")
os.makedirs(CURATOR_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# --- Testing mode: use temp directory instead of outputs/ ---
# Set to False (or remove this block) when you want to save permanently
TESTING_MODE = True

if TESTING_MODE:
    _TEMP_DIR = tempfile.mkdtemp(prefix="pest_gen_")
    UPLOAD_DIR = os.path.join(_TEMP_DIR, "uploads")
    FRAMES_DIR = os.path.join(_TEMP_DIR, "frames")
    VIDEOS_DIR = os.path.join(_TEMP_DIR, "videos")
    LABELS_DIR = os.path.join(_TEMP_DIR, "labels")
    CURATOR_DIR = os.path.join(_TEMP_DIR, "curator")
    print(f"TESTING MODE: outputs go to {_TEMP_DIR} (auto-deleted on exit)")
# ---------------------------------------------------------

app = Flask(__name__)
app.secret_key = "synthetic-pest-gen-dev-key"

# Ensure output directories exist
for d in [UPLOAD_DIR, FRAMES_DIR, VIDEOS_DIR, LABELS_DIR]:
    os.makedirs(d, exist_ok=True)


def _background_warm_models():
    start = time.time()
    try:
        print("Metric3D model warmup started...")
        preload_models()
        print(f"Metric3D model warmup complete in {time.time() - start:.1f}s")
    except Exception as e:
        print(f"WARNING: Metric3D model warmup failed: {e}")


# Warm the model in the background so the first generation request is faster.
threading.Thread(target=_background_warm_models, daemon=True).start()

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

# Stores total render time (seconds) keyed by job_id
_render_times = {}


def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        flash("No file selected.")
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "" or not _allowed_file(file.filename):
        flash("Please upload a valid image file (PNG, JPG, BMP, WEBP).")
        return redirect(url_for("index"))

    # Save uploaded image
    filename = file.filename
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    try:
        t0 = time.time()
        result = generate_video(
            save_path,
            frames_root=FRAMES_DIR,
            labels_root=LABELS_DIR,
            videos_root=VIDEOS_DIR,
        )
        _render_times[result["job_id"]] = round(time.time() - t0, 1)
    except Exception as e:
        flash(f"Generation failed: {e}")
        return redirect(url_for("index"))

    return redirect(url_for("results", job_id=result["job_id"]))


@app.route("/results/<job_id>")
def results(job_id):
    video_path = os.path.join(VIDEOS_DIR, f"{job_id}.mp4")
    frames_dir = os.path.join(FRAMES_DIR, job_id)
    labels_dir = os.path.join(LABELS_DIR, job_id)
    depth_preview_path = os.path.join(frames_dir, "depth_preview.jpg")
    surface_preview_path = os.path.join(frames_dir, "surface_preview.jpg")
    gravity_preview_path = os.path.join(frames_dir, "gravity_preview.jpg")
    probability_preview_path = os.path.join(frames_dir, "probability_preview.jpg")

    if not os.path.exists(video_path):
        flash("Job not found.")
        return redirect(url_for("index"))

    all_job_files = sorted(os.listdir(frames_dir))
    frames = sorted(
        f for f in all_job_files if f.startswith("frame_") and f.endswith(".png")
    )
    placement_masks = sorted(
        f for f in all_job_files if f.startswith("placement_mask_") and f.endswith(".png")
    )

    # Load COCO annotations.json and reorganise by filename for the frontend
    coco_data = {}
    ann_path = os.path.join(labels_dir, "annotations.json")
    if os.path.exists(ann_path):
        with open(ann_path) as f:
            coco_data = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in coco_data.get("images", [])}
    labels_by_frame = {}
    for ann in coco_data.get("annotations", []):
        fname = id_to_filename.get(ann["image_id"])
        if fname:
            labels_by_frame.setdefault(fname, []).append(ann)

    categories_by_id = {
        str(cat["id"]): cat["name"] for cat in coco_data.get("categories", [])
    }

    render_width = 640
    render_height = 480
    if coco_data.get("images"):
        render_width = coco_data["images"][0].get("width", 640)
        render_height = coco_data["images"][0].get("height", 480)

    return render_template(
        "index.html",
        job_id=job_id,
        video_url=url_for("serve_video", job_id=job_id),
        depth_preview_url=(
            url_for("serve_frame", job_id=job_id, filename="depth_preview.jpg")
            if os.path.exists(depth_preview_path)
            else None
        ),
        surface_preview_url=(
            url_for("serve_frame", job_id=job_id, filename="surface_preview.jpg")
            if os.path.exists(surface_preview_path)
            else None
        ),
        gravity_preview_url=(
            url_for("serve_frame", job_id=job_id, filename="gravity_preview.jpg")
            if os.path.exists(gravity_preview_path)
            else None
        ),
        probability_preview_url=(
            url_for("serve_frame", job_id=job_id, filename="probability_preview.jpg")
            if os.path.exists(probability_preview_path)
            else None
        ),
        placement_masks=[
            {
                "filename": fname,
                "url": url_for("serve_frame", job_id=job_id, filename=fname),
                "label": fname.replace("placement_mask_", "").replace(".png", ""),
            }
            for fname in placement_masks
        ],
        frames=frames,
        labels_json=json.dumps(labels_by_frame),
        categories_json=json.dumps(categories_by_id),
        render_width=render_width,
        render_height=render_height,
        render_time=_render_times.get(job_id),
    )


@app.route("/outputs/videos/<job_id>.mp4")
def serve_video(job_id):
    return send_from_directory(VIDEOS_DIR, f"{job_id}.mp4")


@app.route("/outputs/frames/<job_id>/<filename>")
def serve_frame(job_id, filename):
    return send_from_directory(os.path.join(FRAMES_DIR, job_id), filename)


@app.route("/cleanup")
def cleanup():
    """Delete all generated outputs (testing convenience)."""
    for d in [FRAMES_DIR, VIDEOS_DIR, LABELS_DIR, UPLOAD_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
    flash("All outputs cleared.")
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Curator routes
# ---------------------------------------------------------------------------

_PEST_ICONS = {"mouse": "🐭", "rat": "🐀", "cockroach": "🪳"}
_PEST_TYPES = ["mouse", "rat", "cockroach"]

# --- Background pipeline queue (serialises pipeline runs) ---
# _pipeline_busy: True while a job is in-flight for that pest type.
# _pipeline_n_new: n_new requests that arrived while the job was running;
#   the worker re-queues them automatically when the current job finishes.
_pipeline_busy: dict[str, bool] = {p: False for p in _PEST_TYPES}
_pipeline_n_new: dict[str, int] = {p: 0 for p in _PEST_TYPES}
_pipeline_lock = threading.Lock()
_bg_queue: queue.Queue = queue.Queue()


def _bg_pipeline_worker() -> None:
    while True:
        pest_type, n_new = _bg_queue.get()
        try:
            keys = [k for k, v in SPECIES_INFO.items() if v["pest_type"] == pest_type]

            # Count existing candidates per taxon key
            existing_per_key: dict[str, int] = {k: 0 for k in keys}
            for cand in get_curator_status(CURATOR_DIR):
                if cand["taxon_key"] in existing_per_key:
                    existing_per_key[cand["taxon_key"]] += 1

            # Distribute n_new across keys: always give the next image to the
            # key that currently has the fewest candidates.
            targets = dict(existing_per_key)
            for _ in range(n_new):
                k = min(keys, key=lambda k: targets[k])
                targets[k] += 1

            for key in keys:
                if targets[key] > existing_per_key[key]:
                    try:
                        run_pipeline_for_taxon(
                            key, CURATOR_DIR,
                            n_images=targets[key],
                            pest_type=SPECIES_INFO[key]["pest_type"],
                            discards_path=DISCARDS_PATH,
                        )
                    except Exception:
                        traceback.print_exc()
        finally:
            with _pipeline_lock:
                # If more work arrived while we were running, re-queue it now.
                leftover = _pipeline_n_new[pest_type]
                if leftover > 0:
                    _pipeline_n_new[pest_type] = 0
                    _bg_queue.put((pest_type, leftover))
                    # _pipeline_busy stays True — another job is queued
                else:
                    _pipeline_busy[pest_type] = False
            _bg_queue.task_done()


threading.Thread(target=_bg_pipeline_worker, daemon=True, name="bg-pipeline").start()


def _pest_card_summary() -> list[dict]:
    """Return summary dicts for the three pest-type cards."""
    all_candidates = get_curator_status(CURATOR_DIR)
    candidate_counts: dict[str, int] = {}
    for c in all_candidates:
        pest = SPECIES_INFO.get(c["taxon_key"], {}).get("pest_type", "")
        candidate_counts[pest] = candidate_counts.get(pest, 0) + 1

    cards = []
    for pest_type in _PEST_TYPES:
        pest_dir = os.path.join(MODELS_DIR, pest_type)
        kept = len([f for f in os.listdir(pest_dir) if f.endswith(".glb")]) if os.path.isdir(pest_dir) else 0
        cards.append({
            "pest_type": pest_type,
            "icon": _PEST_ICONS[pest_type],
            "kept_count": kept,
            "candidate_count": candidate_counts.get(pest_type, 0),
        })
    return cards


@app.route("/curator")
def curator():
    """Render the model curator page (3 pest-type cards)."""
    return render_template(
        "curator.html",
        pest_cards=_pest_card_summary(),
    )


@app.route("/curator/api/candidates/<pest_type>")
def curator_api_candidates(pest_type):
    """Return all ready candidates for a pest type as JSON, with species metadata."""
    if pest_type not in _PEST_TYPES:
        return jsonify({"status": "error", "error": f"Unknown pest_type: {pest_type}"}), 400

    all_candidates = get_curator_status(CURATOR_DIR)
    result = []
    for c in all_candidates:
        info = SPECIES_INFO.get(c["taxon_key"], {})
        if info.get("pest_type") != pest_type:
            continue
        result.append({
            **c,
            "original_url": "/curator/outputs/{}/{}".format(c["taxon_key"], c["original_url"]),
            "nobg_url":     "/curator/outputs/{}/{}".format(c["taxon_key"], c["nobg_url"]),
            "model_url":    "/curator/outputs/{}/{}".format(c["taxon_key"], c["model_url"]),
            "scientific_name":   info.get("scientific_name", ""),
            "common_names":      info.get("common_names", []),
            "indoor_prevalence": info.get("indoor_prevalence", ""),
            "gbif_url":          info.get("gbif_url", ""),
        })

    # Kept models for this pest type
    pest_dir = os.path.join(MODELS_DIR, pest_type)
    kept = sorted(f for f in os.listdir(pest_dir) if f.endswith(".glb")) if os.path.isdir(pest_dir) else []

    return jsonify({
        "status": "ok",
        "pest_type": pest_type,
        "candidates": result,
        "kept_count": len(kept),
        "kept_models": kept,
    })


@app.route("/curator/run", methods=["POST"])
def curator_run():
    """Run download + rembg + ellipsoid GLB pipeline.

    Body JSON: {"pest_type": "mouse"} | {"taxon_key": "7429082"} | {"taxon_key": "all"}
    Returns JSON with results per taxon.
    """
    data = request.get_json(force=True, silent=True) or {}

    pest_type_param = data.get("pest_type", "")
    taxon_key_param = data.get("taxon_key", "")

    if pest_type_param in _PEST_TYPES:
        keys_to_run = [k for k, v in SPECIES_INFO.items() if v["pest_type"] == pest_type_param]
    elif taxon_key_param == "all":
        keys_to_run = list(SPECIES_INFO.keys())
    elif taxon_key_param in SPECIES_INFO:
        keys_to_run = [taxon_key_param]
    else:
        return jsonify({"status": "error", "error": "Provide pest_type or taxon_key"}), 400

    # Existing candidate counts per taxon. We increase by +2 each run.
    existing_counts: dict[str, int] = {}
    for cand in get_curator_status(CURATOR_DIR):
        key = cand["taxon_key"]
        existing_counts[key] = existing_counts.get(key, 0) + 1

    all_results: dict[str, list[dict]] = {}
    for key in keys_to_run:
        try:
            target_total = existing_counts.get(key, 0) + 2
            results = run_pipeline_for_taxon(
                key,
                CURATOR_DIR,
                n_images=target_total,
                pest_type=SPECIES_INFO[key]["pest_type"],
                discards_path=DISCARDS_PATH,
            )
            all_results[key] = results
        except Exception as e:
            print("curator_run error for taxon", key)
            traceback.print_exc()
            all_results[key] = [{"error": str(e)}]

    return jsonify({"status": "ok", "results": all_results})


@app.route("/curator/keep", methods=["POST"])
def curator_keep():
    """Copy chosen .glb to generator/models/ and patch config.py.

    Body JSON: {"taxon_key": "7429082", "candidate_index": 0}
    """
    data = request.get_json(force=True, silent=True) or {}
    taxon_key = data.get("taxon_key", "")
    candidate_index = data.get("candidate_index")

    if not taxon_key or candidate_index is None:
        return jsonify({"status": "error", "error": "taxon_key and candidate_index required"}), 400

    result = keep_candidate(
        taxon_key=taxon_key,
        candidate_index=int(candidate_index),
        curator_dir=CURATOR_DIR,
        models_dir=MODELS_DIR,
        config_path=CONFIG_PATH,
        species_info=SPECIES_INFO,
    )
    if result["status"] == "error":
        return jsonify(result), 400
    return jsonify(result)


@app.route("/curator/discard", methods=["POST"])
def curator_discard():
    """Delete candidate files and log its GBIF URL to discards.json.

    Body JSON: {"taxon_key": "7429082", "candidate_index": 2}
    discards.json is git-tracked so the URL is never re-downloaded.
    """
    data = request.get_json(force=True, silent=True) or {}
    taxon_key = data.get("taxon_key", "")
    candidate_index = data.get("candidate_index")

    if not taxon_key or candidate_index is None:
        return jsonify({"status": "error", "error": "taxon_key and candidate_index required"}), 400

    result = discard_candidate(
        taxon_key=taxon_key,
        candidate_index=int(candidate_index),
        curator_dir=CURATOR_DIR,
        discards_path=DISCARDS_PATH,
    )
    return jsonify(result)


@app.route("/curator/outputs/<taxon_key>/<filename>")
def curator_serve_output(taxon_key, filename):
    """Serve images and .glb files from outputs/curator/<taxon_key>/."""
    taxon_dir = os.path.join(CURATOR_DIR, taxon_key)
    return send_from_directory(taxon_dir, filename)


@app.route("/curator/models/<filename>")
def curator_serve_model(filename):
    """Serve .glb files from generator/models/."""
    return send_from_directory(MODELS_DIR, filename)


@app.route("/curator/run-async", methods=["POST"])
def curator_run_async():
    """Queue a pipeline for a pest type; returns immediately (fire-and-forget).

    Body JSON: {"pest_type": "mouse", "n_new": 5}
    n_new: how many new images to fetch (default 1).
    If a job is already running, the n_new is accumulated and processed when
    the current job finishes (taking the max of pending values).
    """
    data = request.get_json(force=True, silent=True) or {}
    pest_type = data.get("pest_type", "")
    n_new = max(1, int(data.get("n_new", 1)))
    if pest_type not in _PEST_TYPES:
        return jsonify({"status": "error", "error": f"Unknown pest_type: {pest_type}"}), 400
    with _pipeline_lock:
        # Always take the max so a larger request wins over a smaller one.
        _pipeline_n_new[pest_type] = max(_pipeline_n_new[pest_type], n_new)
        if _pipeline_busy[pest_type]:
            return jsonify({"status": "already_queued"})
        # Not busy: consume the pending n_new and start a job.
        _pipeline_busy[pest_type] = True
        n = _pipeline_n_new[pest_type]
        _pipeline_n_new[pest_type] = 0
        _bg_queue.put((pest_type, n))
    return jsonify({"status": "queued"})


@app.route("/curator/api/pipeline-status")
def curator_pipeline_status():
    """Return {pest: bool} showing which pipelines are queued/running."""
    with _pipeline_lock:
        running = dict(_pipeline_busy)
    return jsonify({"status": "ok", "running": running})


if __name__ == "__main__":
    try:
        app.run(debug=True, host="0.0.0.0", port=5000)
    finally:
        if TESTING_MODE:
            shutil.rmtree(_TEMP_DIR, ignore_errors=True)
            print(f"Cleaned up temp dir: {_TEMP_DIR}")
