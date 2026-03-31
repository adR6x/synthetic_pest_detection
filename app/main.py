"""Flask web application — upload kitchen images and generate synthetic pest videos."""

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import threading
import time

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash

from generator.config import UPLOAD_DIR, FRAMES_DIR, VIDEOS_DIR, LABELS_DIR
from generator.depth_estimator import (
    preload_models,
    estimate_metric3d,
    save_depth_preview,
    save_surface_preview,
)
from generator.kitchen_img.download_kitchens import (
    OUT_DIR as KITCHEN_IMG_DIR,
    extract_places_filename,
    list_local_kitchen_images,
    mark_places_as_seen,
)
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

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
CURATOR_PREVIEW_DIR = os.path.join(KITCHEN_IMG_DIR, ".curator_cache")
CURATED_PAGE_SIZE = 5
os.makedirs(CURATOR_PREVIEW_DIR, exist_ok=True)


# Stores total render time (seconds) keyed by job_id
_render_times = {}
# Stores the source image path used for each job so regeneration doesn't need a re-upload
_source_images = {}


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


def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _parse_positive_int(value, default=1):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _get_curated_page(page, page_size=CURATED_PAGE_SIZE):
    images = list_local_kitchen_images()
    total = len(images)
    if total == 0:
        return {
            "images": [],
            "total": 0,
            "page": 1,
            "total_pages": 1,
            "has_prev": False,
            "has_next": False,
            "prev_page": 1,
            "next_page": 1,
        }

    total_pages = max(1, math.ceil(total / page_size))
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    page_images = images[start:start + page_size]
    return {
        "images": page_images,
        "total": total,
        "page": page,
        "total_pages": total_pages,
        "has_prev": page > 1,
        "has_next": page < total_pages,
        "prev_page": page - 1 if page > 1 else 1,
        "next_page": page + 1 if page < total_pages else total_pages,
    }


def _generate_video_for_image(image_path):
    t0 = time.time()
    result = generate_video(
        image_path,
        frames_root=FRAMES_DIR,
        labels_root=LABELS_DIR,
        videos_root=VIDEOS_DIR,
    )
    _render_times[result["job_id"]] = round(time.time() - t0, 1)
    _source_images[result["job_id"]] = image_path
    return result


def _render_generate_page(job_context=None):
    page = _parse_positive_int(request.args.get("page", 1), default=1)
    curated = _get_curated_page(page)
    context = {
        "active_tab": "generate",
        "curated_page_images": curated["images"],
        "curated_page": curated["page"],
        "curated_total": curated["total"],
        "curated_total_pages": curated["total_pages"],
        "curated_has_prev": curated["has_prev"],
        "curated_has_next": curated["has_next"],
        "curated_prev_page": curated["prev_page"],
        "curated_next_page": curated["next_page"],
    }
    if job_context:
        context.update(job_context)
    return render_template("index.html", **context)


def _curator_preview_names(filename):
    digest = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:12]
    stem = os.path.splitext(filename)[0]
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)[:60]
    base = f"{safe_stem}_{digest}"
    return f"{base}_depth.jpg", f"{base}_surface.jpg"


def _ensure_curator_scene_analysis(filename):
    source_path = os.path.join(KITCHEN_IMG_DIR, filename)
    if not os.path.exists(source_path):
        return None, None

    depth_name, surface_name = _curator_preview_names(filename)
    depth_path = os.path.join(CURATOR_PREVIEW_DIR, depth_name)
    surface_path = os.path.join(CURATOR_PREVIEW_DIR, surface_name)

    source_mtime = os.path.getmtime(source_path)
    cache_fresh = (
        os.path.exists(depth_path)
        and os.path.exists(surface_path)
        and os.path.getmtime(depth_path) >= source_mtime
        and os.path.getmtime(surface_path) >= source_mtime
    )

    if not cache_fresh:
        metric3d_result = estimate_metric3d(source_path)
        save_depth_preview(metric3d_result["depth"], depth_path)
        save_surface_preview(metric3d_result["normals"], surface_path)

    return depth_name, surface_name


@app.route("/", methods=["GET"])
def index():
    return _render_generate_page()


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
    page = _parse_positive_int(request.form.get("page", 1), default=1)

    try:
        result = _generate_video_for_image(save_path)
    except Exception as e:
        flash(f"Generation failed: {e}")
        return redirect(url_for("index", page=page))

    return redirect(url_for("results", job_id=result["job_id"], page=page))


@app.route("/generate/curated", methods=["POST"])
def generate_from_curated():
    filename = (request.form.get("filename") or "").strip()
    page = _parse_positive_int(request.form.get("page", 1), default=1)

    images = list_local_kitchen_images()
    if not filename or filename not in images:
        flash("Selected curated image was not found.")
        return redirect(url_for("index", page=page))

    image_path = os.path.join(KITCHEN_IMG_DIR, filename)
    if not os.path.exists(image_path):
        flash(f"Image file not found: {filename}")
        return redirect(url_for("index", page=page))

    try:
        result = _generate_video_for_image(image_path)
    except Exception as e:
        flash(f"Generation failed for {filename}: {e}")
        return redirect(url_for("index", page=page))

    return redirect(url_for("results", job_id=result["job_id"], page=page))


@app.route("/results/<job_id>")
def results(job_id):
    video_path = os.path.join(VIDEOS_DIR, f"{job_id}.mp4")
    frames_dir = os.path.join(FRAMES_DIR, job_id)
    labels_dir = os.path.join(LABELS_DIR, job_id)
    depth_preview_path = os.path.join(frames_dir, "depth_preview.jpg")
    surface_preview_path = os.path.join(frames_dir, "surface_preview.jpg")
    gravity_preview_path = os.path.join(frames_dir, "gravity_preview.jpg")

    if not os.path.exists(video_path):
        flash("Job not found.")
        return redirect(url_for("index"))

    all_job_files = sorted(os.listdir(frames_dir))
    frames = sorted(
        f for f in all_job_files if f.startswith("frame_") and f.endswith(".png")
    )

    # Build per-pest mask preview URL templates.
    # Files are named: mask_preview_pest{i}_{pesttype}_{frame:04d}.png
    # We scan for frame 0001 to discover which pests exist.
    _pest_mask_re = re.compile(r"^mask_preview_(pest(\d+)_(\w+))_0001\.png$")
    pest_mask_templates = []
    for fname in all_job_files:
        m = _pest_mask_re.match(fname)
        if m:
            base, pest_idx, pest_type = m.group(1), int(m.group(2)), m.group(3)
            pest_mask_templates.append({
                "pest_idx": pest_idx,
                "label": f"{pest_type.capitalize()} #{pest_idx}",
                "url_template": url_for(
                    "serve_frame", job_id=job_id,
                    filename=f"mask_preview_{base}_NNNN.png"
                ),
            })
    pest_mask_templates.sort(key=lambda t: t["pest_idx"])

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

    return _render_generate_page({
        "job_id": job_id,
        "video_url": url_for("serve_video", job_id=job_id),
        "depth_preview_url": (
            url_for("serve_frame", job_id=job_id, filename="depth_preview.jpg")
            if os.path.exists(depth_preview_path)
            else None
        ),
        "surface_preview_url": (
            url_for("serve_frame", job_id=job_id, filename="surface_preview.jpg")
            if os.path.exists(surface_preview_path)
            else None
        ),
        "gravity_preview_url": (
            url_for("serve_frame", job_id=job_id, filename="gravity_preview.jpg")
            if os.path.exists(gravity_preview_path)
            else None
        ),
        "pest_mask_templates": pest_mask_templates,
        "frames": frames,
        "labels_json": json.dumps(labels_by_frame),
        "categories_json": json.dumps(categories_by_id),
        "render_width": render_width,
        "render_height": render_height,
        "render_time": _render_times.get(job_id),
    })


@app.route("/curator", methods=["GET"])
def curator():
    try:
        idx = int(request.args.get("idx", 0))
    except ValueError:
        idx = 0

    images = list_local_kitchen_images()
    total = len(images)

    current = None
    depth_preview_url = None
    surface_preview_url = None
    analysis_error = None

    if total > 0:
        idx = max(0, min(idx, total - 1))
        current = images[idx]
        try:
            depth_name, surface_name = _ensure_curator_scene_analysis(current)
            if depth_name and surface_name:
                depth_preview_url = url_for("serve_kitchen_preview", filename=depth_name)
                surface_preview_url = url_for("serve_kitchen_preview", filename=surface_name)
        except Exception as e:
            analysis_error = f"Scene analysis failed for {current}: {e}"
    else:
        idx = 0

    return render_template(
        "index.html",
        active_tab="curator",
        curator_images=images,
        curator_current=current,
        curator_idx=idx,
        curator_total=total,
        curator_depth_preview_url=depth_preview_url,
        curator_surface_preview_url=surface_preview_url,
        curator_analysis_error=analysis_error,
    )


@app.route("/curator/action", methods=["POST"])
def curator_action():
    action = (request.form.get("action") or "").strip().lower()
    filename = (request.form.get("filename") or "").strip()

    try:
        idx = int(request.form.get("idx", 0))
    except ValueError:
        idx = 0

    images = list_local_kitchen_images()
    if not filename or filename not in images:
        flash("Selected kitchen image was not found.")
        return redirect(url_for("curator", idx=idx))

    places_name = extract_places_filename(filename)
    if places_name:
        mark_places_as_seen({places_name})

    if action == "keep":
        flash(f"Kept {filename}")
        next_idx = min(idx + 1, max(0, len(images) - 1))
        return redirect(url_for("curator", idx=next_idx))

    if action == "delete":
        path = os.path.join(KITCHEN_IMG_DIR, filename)
        try:
            os.remove(path)
        except OSError as e:
            flash(f"Failed to delete {filename}: {e}")
            return redirect(url_for("curator", idx=idx))

        flash(f"Deleted {filename}")
        remaining = list_local_kitchen_images()
        if not remaining:
            return redirect(url_for("curator"))
        next_idx = min(idx, len(remaining) - 1)
        return redirect(url_for("curator", idx=next_idx))

    flash("Unknown action.")
    return redirect(url_for("curator", idx=idx))


@app.route("/regenerate/<job_id>", methods=["POST"])
def regenerate(job_id):
    image_path = _source_images.get(job_id)
    if not image_path or not os.path.exists(image_path):
        flash("Source image no longer available — please upload again.")
        return redirect(url_for("index"))

    try:
        t0 = time.time()
        result = generate_video(
            image_path,
            frames_root=FRAMES_DIR,
            labels_root=LABELS_DIR,
            videos_root=VIDEOS_DIR,
        )
        _render_times[result["job_id"]] = round(time.time() - t0, 1)
        _source_images[result["job_id"]] = image_path
    except Exception as e:
        flash(f"Regeneration failed: {e}")
        return redirect(url_for("results", job_id=job_id))

    return redirect(url_for("results", job_id=result["job_id"]))


@app.route("/outputs/videos/<job_id>.mp4")
def serve_video(job_id):
    return send_from_directory(VIDEOS_DIR, f"{job_id}.mp4")


@app.route("/outputs/frames/<job_id>/<filename>")
def serve_frame(job_id, filename):
    return send_from_directory(os.path.join(FRAMES_DIR, job_id), filename)


@app.route("/kitchen-images/<path:filename>")
def serve_kitchen_image(filename):
    return send_from_directory(KITCHEN_IMG_DIR, filename)


@app.route("/kitchen-previews/<path:filename>")
def serve_kitchen_preview(filename):
    return send_from_directory(CURATOR_PREVIEW_DIR, filename)


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
