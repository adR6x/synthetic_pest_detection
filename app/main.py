"""Flask web application — upload kitchen images and generate synthetic pest videos."""

import json
import os
import shutil
import tempfile
import threading
import time

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash

from generator.config import UPLOAD_DIR, FRAMES_DIR, VIDEOS_DIR, LABELS_DIR, OUTPUT_DIR
from generator.depth_estimator import preload_models
from generator.pipeline import generate_video

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

    # Load per-frame labels
    labels_by_frame = {}
    for frame_name in frames:
        label_name = frame_name.replace(".png", ".json")
        label_path = os.path.join(labels_dir, label_name)
        if os.path.exists(label_path):
            with open(label_path) as f:
                labels_by_frame[frame_name] = json.load(f)

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


if __name__ == "__main__":
    try:
        app.run(debug=True, host="0.0.0.0", port=5000)
    finally:
        if TESTING_MODE:
            shutil.rmtree(_TEMP_DIR, ignore_errors=True)
            print(f"Cleaned up temp dir: {_TEMP_DIR}")
