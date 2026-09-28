# SAM 3 Card and Clothing Measurement

Experiments using Meta's SAM 3 image segmentation model to detect cards and clothing, measure objects from image scale references, and estimate garment dimensions from contours.

## Project contents

- `sam3_source/` — SAM 3 source as a Git submodule pointing to Meta's upstream repository.
- `evaluate_cards.py` — card detection and annotated result generation.
- `measure_object.py` — object measurement using a detected card as a physical-size reference.
- `geometric_contour.py` — garment segmentation and contour-based measurement.
- `test_cards/`, `test_clothes/` — sample input images.
- `card_results/`, `measurement_results/`, `contour_measurement_results/` — sample output images. Running the scripts may overwrite these results.

Other scripts and folders in this workspace, including the separate YOLO clothing models and their checkpoints, are not part of this SAM 3 project.

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

## Upstream project

SAM 3 is developed by Meta. Its source and license are included under `sam3_source/`; see `sam3_source/README.md` and `sam3_source/LICENSE` for upstream details. Model checkpoint access and use are subject to Meta's terms.
