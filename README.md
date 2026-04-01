# Synthetic Data Generation for Pest Detection

Synthetic kitchen-scene video generation for pest detection (mouse, rat, cockroach), with a Flask web app for curation, generation, and dataset prep.

## What The App Supports

The web app has four tabs:

1. `Test Video Generator`
- Generate one synthetic video from a selected kitchen image.
- Shows video + frame overlays + scene analysis previews.

2. `Real Video Generator`
- Batch-generate training jobs from curated kitchens.
- Configurable: length, fps, number of videos, optional MP4 output.
- Progress bar + live polling + 5-row paginated results.
- Writes metadata to `outputs/generated_state.json`.
- Uses [`generator/kitchen_img/test_train_split.csv`](generator/kitchen_img/test_train_split.csv) to assign each curated kitchen to train or test.
- Future real-generation outputs are written under `outputs/train/` or `outputs/test/`.
- Persists sparse training frames named `frame_0001`, `frame_0010`, `frame_0020`, ...
- Keeps full `annotations.json` metadata even when only sparse frame images are saved.
- If MP4 output is enabled, a dense temporary frame set is rendered for smooth video assembly, then only the sparse training frames are kept permanently.

3. `Kitchen Curator`
- Review `uncurated_img/` images.
- `Keep` moves image to `curated_img/` and assigns `kitchen_####.ext` ID.
- `Delete` removes image and marks source as seen.
- Supports downloading more Places365 images.

4. `Kitchen Generator`
- Generate kitchen images with Gemini API.
- Generated images go to `uncurated_img/` first, then can be curated.

## Setup

Requires Python 3.10+ (Python 3.12 recommended).

### macOS / Linux / WSL

```bash
bash setupUNIX.sh
poetry shell
```

### Windows (PowerShell)

```powershell
.\setupPC.ps1
poetry shell
```

Setup scripts install all required Python packages from `pyproject.toml` and install the local `mmcv` compatibility stub used by Metric3D dependencies. On Linux, `setupUNIX.sh` also attempts to install OpenCV runtime libraries (`libgl1`, `libglib2.0-0`) when missing.

## Run The App

Inside `poetry shell`:

```bash
flask --app app.main run
# Open http://localhost:5000
```

## Current Data Layout

```text
outputs/
  uploads/                  # ad-hoc uploads
  frames/{job_id}/          # legacy real/test outputs
  labels/{job_id}/          # legacy COCO annotations.json
  videos/{job_id}.mp4       # legacy optional MP4 outputs
  generated_state.json      # real-batch metadata state
  train/
    frames/{job_id}/        # sparse real-generation frame images
    labels/{job_id}/        # full COCO annotations.json for the job
    videos/{job_id}.mp4     # optional real-generation MP4s
  test/
    frames/{job_id}/
    labels/{job_id}/
    videos/{job_id}.mp4

generator/kitchen_img/
  uncurated_img/            # downloaded or Gemini-generated, pending review
  curated_img/              # approved kitchens, named as kitchen_####.jpg
  download_state.json       # seen Places IDs + kitchen/source mapping
  test_train_split.csv      # kitchen-level 75/25 train/test assignment
```

## State Files

### `outputs/generated_state.json`

Top-level keys:
- `generated_videos`: list of generation records
- `updated_at`

Each generation record currently includes:
- `job_id`
- `video_id`
- `kitchen_id` (filename in curated set, e.g. `kitchen_0016.jpg`)
- `split` (`train` or `test`) for newer real-generation rows
- `train` (`1` for train, `0` for test) for newer real-generation rows
- `length_of_video_seconds`
- `fps`
- `mouse_count`
- `rat_count`
- `cockroach_count`
- `date_time_generated`
- `time_taken_to_generate_seconds`
- `pest_size_multiplier`
- `pest_generation_metadata` (per pest):
  - `pest_index`, `pest_type`
  - `relative_size_scale`
  - `relative_size_image_fraction`
  - `approx_initial_pixel_width`
  - `initial_position_image_px`
  - `initial_position_image_norm`

### `generator/kitchen_img/download_state.json`

Top-level keys:
- `seen_places365_files`
- `kitchen_mappings`

`kitchen_mappings` maps curated `kitchen_id` to Places365 source metadata:
- `places365_source_id`
- `places365_link`
- `linked_at`

## Places365 Download Behavior

- Downloader uses Places365 metadata (`places365_train_standard.txt` or `places365_val.txt`) to find class-203 kitchens.
- It then streams the selected split archive tar and extracts only selected unseen kitchens.
- `val` has 100 kitchen images total; once all are seen, curator download reports no unseen.
- `download_state.json` prevents re-downloading images already seen/curated/deleted.

## Generation Pipeline (Current)

Implemented in `generator/pipeline.py`:

1. Load kitchen image.
2. Run Metric3D + gravity estimation.
3. Build surface-aware spawn/movement masks.
4. Sample number of pests `N` with this distribution:
- `P(N=0)=0.25`
- `P(N=1)=0.30`
- Remaining 0.45 distributed exponentially over `N=2..6`.
5. For each pest slot, sample type with:
- cockroach `0.50`
- mouse `0.30`
- rat `0.20`
6. Compute depth-aware scale and apply global multiplier `1.8x`.
7. Composite frames + COCO labels.
8. Optionally assemble MP4.

## Training Pipeline

### 1) Prepare DETR dataset

```bash
python -m training.prepare_dataset \
  --frames_root outputs/train/frames \
  --labels_root outputs/train/labels \
  --output_dir outputs/dataset \
  --val_frac 0.1 --test_frac 0.1 \
  --every_n 10
```

Current split logic:
- split by **job/video** (not frame)
- default ~80/10/10 train/val/test
- loader filters by frame filenames actually present on disk, so it works with dense legacy outputs and newer sparse-saved outputs
- for newer real-generation outputs, frame images are sparse but `annotations.json` may still contain entries for unsaved frames; this is expected and handled by the loader

### Kitchen Train/Test Split

- `generator/kitchen_img/test_train_split.csv` is the source of truth for kitchen-level train/test assignment.
- It has columns `id,train`, where `id` is the curated kitchen filename and `train` is `1` for train, `0` for test.
- Regenerate it with:

```bash
python3 generator/kitchen_img/test_train_split.py
```

### 2) Train DETR

```bash
python -m training.train --data_dir outputs/dataset --freeze_backbone
```

## Notes

- `curated_img` uses `kitchen_####` naming (legacy `kitchen_img_####` is migrated automatically where applicable).
- Real generation can run with or without MP4 assembly.
- Null cases are supported (`N=0`) in generation.
