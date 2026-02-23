# Synthetic Data Generation for Pest Detection

A Blender-based synthetic video generator that overlays animated 3D pests (mice, rats, cockroaches) onto kitchen images using depth-aware placement, producing labeled frames for training a Vision Transformer classifier.

## Setup

Requires Python 3.12+ and Blender 3.6+ on PATH (Blender only needed for generation).

### macOS / Linux / WSL

```bash
bash setupUNIX.sh   # installs Poetry, shell plugin, and all packages
poetry shell        # activate the environment
```

### Windows (PowerShell)

```powershell
.\setupPC.ps1       # installs Poetry, shell plugin, and all packages
poetry shell        # activate the environment
```

If PowerShell blocks script execution, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setupPC.ps1
```

## Running the App

Inside `poetry shell`:

```bash
flask run
# Open http://localhost:5000
# Upload a kitchen image -> generates video + 10 labeled frames
```

## Training the Classifier

```bash
python -m training.train
# Trains a ViT classifier on generated frames in outputs/
# Prints "no data" if nothing has been generated yet
```

## Project Structure

```
├── app/
│   ├── main.py               # Flask routes: upload, generate, serve results
│   ├── templates/index.html  # Upload form, frame player, bbox overlay, previews
│   └── static/style.css
│
├── generator/
│   ├── config.py             # Render settings, pest parameters, output paths
│   ├── pipeline.py           # Orchestrator: depth estimation + Blender subprocess + MP4 assembly
│   ├── depth_estimator.py    # Depth + surface normal estimation, probability placement maps
│   ├── blender_script.py     # Entry point inside Blender
│   ├── scene_setup.py        # Background plane, orthographic camera, lighting (bpy)
│   ├── pest_models.py        # 3D pest geometry from UV sphere primitives (bpy)
│   ├── pest_animation.py     # Random-walk keyframe animation with mask sampling (bpy)
│   └── labeler.py            # 3D->2D bbox projection + JSON labels (bpy)
│
├── training/
│   ├── config.py             # Hyperparameters, label mapping
│   ├── dataset.py            # PyTorch Dataset (frames + JSON labels)
│   ├── model.py              # ViT-base-patch16-224 wrapper (HuggingFace)
│   └── train.py              # Training loop: AdamW, CrossEntropy, per-epoch logging
│
├── setupUNIX.sh              # One-command setup for macOS/Linux/WSL (Poetry)
├── setupPC.ps1               # One-command setup for Windows PowerShell (Poetry)
├── pyproject.toml            # Poetry dependencies
└── outputs/                  # Generated data (gitignored)
    ├── uploads/
    ├── frames/{job_id}/
    ├── videos/{job_id}.mp4
    └── labels/{job_id}/
```

## Architecture

```
User uploads kitchen.jpg
  -> Flask saves to outputs/uploads/
  -> pipeline.py:
       1. Depth Anything V2  — estimates depth map
       2. Omnidata DPT       — estimates surface normals
       3. Probability map    — sigmoid slope + normal coherence per pest type
       4. Pest sampling      — positions drawn from probability map
       5. Blender subprocess — renders 10 frames at 640x480 via EEVEE
          -> scene_setup.py      background plane, camera, sun light
          -> pest_models.py      ellipsoid meshes
          -> pest_animation.py   random-walk keyframes + mask rejection sampling
          -> labeler.py          3D->2D bbox projection + JSON
       6. OpenCV             — assembles PNGs into MP4 at 2 FPS
  -> Flask serves video + frame gallery + depth/normal/mask previews
```

## Label Format

Per-frame JSON in `outputs/labels/{job_id}/frame_XXXX.json`:

```json
{
  "frame": 1,
  "width": 640,
  "height": 480,
  "annotations": [
    {
      "pest_type": "mouse",
      "bbox": [x_min, y_min, x_max, y_max],
      "confidence": 1.0
    }
  ]
}
```

## Classes

| ID | Label |
|----|-------|
| 0 | background |
| 1 | mouse |
| 2 | rat |
| 3 | cockroach |
