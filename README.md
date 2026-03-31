# Synthetic Data Generation for Pest Detection

A synthetic video generator that overlays animated 2D pest sprites (mouse, rat, cockroach) onto kitchen images using depth-aware placement. It produces frame-level COCO bounding-box labels and MP4 videos for downstream training.

## Setup

Requires Python 3.12+.

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

## Running the App

Inside `poetry shell`:

```bash
flask run
# Open http://localhost:5000
# Upload a kitchen image -> generates video + labeled frames
```

## Training the Classifier

```bash
python -m training.train
# Trains ViT classifier on generated frames in outputs/
```

## Project Structure

```text
├── app/
│   ├── main.py               # Flask routes: upload, generate, serve results
│   ├── templates/index.html  # Upload form, frame player, bbox overlay, previews
│   └── static/style.css
│
├── generator/
│   ├── config.py             # Render settings, pest params, paths
│   ├── pipeline.py           # Orchestrator: depth/normal + placement + compositing + MP4 assembly
│   ├── depth_estimator.py    # Metric3D v2 + gravity estimation + probability maps
│   ├── compositing.py        # 2D sprite renderer + COCO bbox generation
│   ├── pest_models.py        # Sprite loader; PNG sprites or procedural fallback
│   ├── pest_animation.py     # Random-walk trajectory generation
│   ├── labeler.py            # COCO JSON writer
│   ├── sprites/
│   │   ├── mouse/            # Optional custom RGBA PNG sprites
│   │   ├── rat/
│   │   └── cockroach/
│   └── mmcv_stub/            # Minimal mmcv shim for Metric3D dependency chain
│
├── training/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   └── train.py
│
├── setupUNIX.sh
├── setupPC.ps1
├── pyproject.toml
└── outputs/
    ├── uploads/
    ├── frames/{job_id}/
    ├── videos/{job_id}.mp4
    └── labels/{job_id}/annotations.json
```

## Architecture

```text
User uploads kitchen.jpg
  -> Flask saves to outputs/uploads/
  -> pipeline.py:
       1. Feed-forward inference:
          a. Metric3D v2 (ViT-small): metric depth + surface normals
          b. Gravity estimation via vanishing-point detection
       2. Build per-pest placement probability map from normals
       3. Sample pest start positions from placement map
       4. Compute depth-aware body size in world units
       5. composite_frames():
          -> compute random-walk paths
          -> load sprite (PNG from generator/sprites/* if present)
          -> fallback to procedural shape if no sprite asset exists
          -> resize/rotate/paste sprite per frame
          -> emit COCO bbox annotations
       6. ffmpeg (H.264) assembly to MP4 (OpenCV mp4v fallback)
  -> Flask serves video + frame gallery + previews
```

## Sprite Workflow

By default, sprite folders are empty and the pipeline uses procedural sprites generated from `PEST_PARAMS`.

### Default fallback behavior

- Sprite lookup path: `generator/sprites/{pest_type}/*.png`
- If none found, the renderer draws a procedural top-down pest sprite (ellipse/body + head for mouse/rat).
- Procedural fallback is valid for generation and training bootstrapping.

### Adding custom sprites

1. Create RGBA PNG(s) with transparent background.
2. Drop files into:
   - `generator/sprites/mouse/`
   - `generator/sprites/rat/`
   - `generator/sprites/cockroach/`
3. Re-run generation. A random sprite per pest type is selected per sample.

Notes:
- Sprite forward direction is assumed to be +X (head pointing right) unless you change `PEST_FORWARD_AXIS`.
- Size is depth-aware; sprite pixel width derives from estimated world size and scene scale.

## Depth-Based Size Scaling

The pipeline computes a depth-aware world-size term per pest:

```text
scale = real_body_length_m * fx * PLANE_WIDTH / (depth_m * RENDER_WIDTH)
```

Then compositing maps that world size to pixel width:

```text
pixel_w = scale * RENDER_WIDTH / PLANE_WIDTH
```

This keeps pests larger when closer and smaller when farther, while retaining stable class-specific physical priors.

## Annotation Format

A single COCO-format JSON is written to `outputs/labels/{job_id}/annotations.json`.

```json
{
  "images": [
    {"id": 1, "file_name": "frame_0001.png", "width": 640, "height": 480}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": width * height,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "mouse", "supercategory": "pest"},
    {"id": 2, "name": "rat", "supercategory": "pest"},
    {"id": 3, "name": "cockroach", "supercategory": "pest"}
  ]
}
```

Current bbox behavior:
- Bboxes are axis-aligned rectangles from the pasted rotated sprite bounds (clipped to frame).
- They are not alpha-tight pixel masks.
- Segmentation polygons/masks and `track_id` are not currently emitted.

## Models

### Depth + Surface Normals — Metric3D v2 (ViT-small)

Single forward pass yields geometrically consistent metric depth and normals.

Paper: Hu et al., *Metric3D v2: A Versatile Monocular Geometric Foundation Model for Zero-Shot Metric Depth and Surface Normal Estimation*, IEEE TPAMI 2024.
[arXiv:2404.15506](https://arxiv.org/abs/2404.15506)

### Gravity Estimation — Classical Vanishing Point

Uses OpenCV line segments + RANSAC to estimate camera up-vector for scene interpretation.

## Classes (ViT Classifier)

`training/config.py` mapping:

| ID | Label |
|----|-------|
| 0  | background |
| 1  | mouse |
| 2  | rat |
| 3  | cockroach |
