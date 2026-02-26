"""Flask web application — upload kitchen images and generate synthetic pest videos."""

import json
import os
import shutil
import tempfile
import threading
import time

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify

from generator.config import UPLOAD_DIR, FRAMES_DIR, VIDEOS_DIR, LABELS_DIR, OUTPUT_DIR, PROJECT_ROOT
from generator.depth_estimator import preload_models
from generator.pipeline import generate_video
from generator.model_curator.species_info import SPECIES_INFO
from generator.model_curator.curator import (
    get_curator_status,
    run_pipeline_for_taxon,
    keep_candidate,
    TRIPOSR_AVAILABLE,
)

# --- Curator constants ---
CURATOR_DIR = os.path.join(PROJECT_ROOT, "outputs", "curator")
MODELS_DIR = os.path.join(PROJECT_ROOT, "generator", "models")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "generator", "config.py")
os.makedirs(CURATOR_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def _check_triposr() -> bool:
    try:
        import tsr  # noqa: F401
        return True
    except ImportError:
        return False

# --- Testing mode: use temp directory instead of outputs/ ---
# Set to False (or remove this block) when you want to save permanently
TESTING_MODE = True

if TESTING_MODE:
    _TEMP_DIR = tempfile.mkdtemp(prefix="pest_gen_")
    UPLOAD_DIR = os.path.join(_TEMP_DIR, "uploads")
    FRAMES_DIR = os.path.join(_TEMP_DIR, "frames")
    VIDEOS_DIR = os.path.join(_TEMP_DIR, "videos")
    LABELS_DIR = os.path.join(_TEMP_DIR, "labels")
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

@app.route("/curator")
def curator():
    """Render the model curator page."""
    # Group existing candidates by taxon_key
    all_candidates = get_curator_status(CURATOR_DIR)
    candidates_by_taxon: dict[str, list[dict]] = {}
    for c in all_candidates:
        candidates_by_taxon.setdefault(c["taxon_key"], []).append(c)

    return render_template(
        "curator.html",
        species_info=SPECIES_INFO,
        candidates_by_taxon=candidates_by_taxon,
        triposr_available=_check_triposr(),
    )


@app.route("/curator/run", methods=["POST"])
def curator_run():
    """Run download + rembg + (optional) TripoSR pipeline.

    Body JSON: {"taxon_key": "7429082"} or {"taxon_key": "all"}
    Returns JSON with results per taxon.
    """
    data = request.get_json(force=True, silent=True) or {}
    taxon_key_param = data.get("taxon_key", "")
    triposr = _check_triposr()

    if taxon_key_param == "all":
        keys_to_run = list(SPECIES_INFO.keys())
    elif taxon_key_param in SPECIES_INFO:
        keys_to_run = [taxon_key_param]
    else:
        return jsonify({"status": "error", "error": f"Unknown taxon_key: {taxon_key_param}"}), 400

    all_results: dict[str, list[dict]] = {}
    for key in keys_to_run:
        try:
            results = run_pipeline_for_taxon(
                key,
                CURATOR_DIR,
                n_images=5,
                triposr_available=triposr,
            )
            all_results[key] = results
        except Exception as e:
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


@app.route("/curator/outputs/<taxon_key>/<filename>")
def curator_serve_output(taxon_key, filename):
    """Serve images and .glb files from outputs/curator/<taxon_key>/."""
    taxon_dir = os.path.join(CURATOR_DIR, taxon_key)
    return send_from_directory(taxon_dir, filename)


@app.route("/curator/models/<filename>")
def curator_serve_model(filename):
    """Serve .glb files from generator/models/."""
    return send_from_directory(MODELS_DIR, filename)


if __name__ == "__main__":
    try:
        app.run(debug=True, host="0.0.0.0", port=5000)
    finally:
        if TESTING_MODE:
            shutil.rmtree(_TEMP_DIR, ignore_errors=True)
            print(f"Cleaned up temp dir: {_TEMP_DIR}")
