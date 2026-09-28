# SAM 3 Card and Clothing Measurement

Experiments using Meta's SAM 3 image segmentation model to detect cards and clothing, measure objects from image scale references, and estimate garment dimensions from contours.

## Project contents

- `sam3_source/` — SAM 3 source as a Git submodule pointing to Meta's upstream repository.
- `evaluate_cards.py` — card detection script for testing.
- `measure_object.py` — object measurement using a detected card as a physical-size reference.
- `geometric_contour.py` — garment segmentation and contour-based measurement.
- `test_cards/`, `test_clothes/` — sample input images.
- `card_results/`, `measurement_results/`, `contour_measurement_results/` — sample output images. Running the scripts may overwrite these results.

Clone this repository with its SAM 3 source:

```bash
git clone --recurse-submodules https://github.com/ashwinjmonteiro/GarmentSense.git
cd GarmentSense
```

If you already cloned without submodules, initialize it with `git submodule update --init --recursive`.

## Requirements

- Python 3.10 or newer
- PyTorch and torchvision installed for your operating system and CPU/CUDA setup (follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/)).
- Access to the SAM 3 checkpoint on Hugging Face. Follow the access instructions in the [SAM 3 repository](https://github.com/facebookresearch/sam3); authenticate with Hugging Face if required. The checkpoint is downloaded by SAM 3 and is not stored in this repository.

The scripts also use OpenCV, NumPy, and Pillow. From the project root, install SAM 3 and the image dependencies in your activated virtual environment:

```powershell
python -m pip install -e .\sam3_source
python -m pip install opencv-python pillow
```

```bash
python -m pip install -e ./sam3_source
python -m pip install opencv-python pillow
```

Install PyTorch before these steps. The SAM 3 package declares its other Python dependencies in `sam3_source/pyproject.toml`.

## Run

Run commands from the project root so the relative input and output paths resolve correctly:

```bash
python evaluate_cards.py
python measure_object.py
python geometric_contour.py
```

Each script's `__main__` section selects a sample input. Edit that path or call its measurement function with another image to process a different sample. Results are written to the corresponding output directory.

## Measurement notes

Card measurements use standard physical card dimensions as a scale reference. Clothing measurements and size estimates depend on the detected garment contour, image orientation, and the assumptions encoded in `geometric_contour.py`; treat them as estimates rather than calibrated measurements.

## Resolution and distance validation

The minimum usable image resolution and maximum working distance have not yet been established. The example images do not record controlled camera distances or reference measurements, so they are not enough to support those limits. Distance also depends on the camera, lens, focus, lighting, and the number of pixels covering both the garment and calibration card.

Use a fixed camera and the same garment, card, background, and lighting for each capture. Keep the card beside the garment and in the same plane. Record the camera/device, image dimensions, distance from camera to the objects, and manually measured garment dimensions. Keep focus, zoom, and other camera settings fixed where possible.

1. Capture a full-resolution image at a close, clear distance and confirm that the card and garment are detected and that the landmarks are sensible.
2. Test lower resolutions by downsampling that same image, preserving its field of view. Record the image dimensions and the pixel width/height of the card and garment. This isolates resolution from changes in pose, lighting, and framing.
3. Restore the original resolution and capture new images at increasing measured distances. Do not simulate distance by resizing: moving the camera also changes perspective, focus, and lighting.
4. For every image, record whether both detections succeeded, whether all required landmarks are present and on the garment, and the error of each reported dimension against a tape-measure reference. Decide the acceptable measurement error before testing.
5. Repeat captures near the first failures. Report the lowest tested resolution and greatest tested distance that meet the chosen criteria consistently, alongside the camera and setup. Also report the corresponding card and garment pixel sizes; these are more transferable than distance alone.

The current scripts print measurements and save overlays, but do not calculate accuracy against reference measurements or summarize a resolution/distance sweep. The limits should be reported only after collecting and reviewing those validation results.

## Upstream project

SAM 3 is developed by Meta. Its source and license are included under `sam3_source/`; see `sam3_source/README.md` and `sam3_source/LICENSE` for upstream details. Model checkpoint access and use are subject to Meta's terms.
