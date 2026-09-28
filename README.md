# SAM 3 Garment and Object Measurement

This project uses Meta's SAM 3 image segmentation model to detect a reference card and garments, extract geometric landmarks from their masks, and estimate measurements. Development has focused on card-calibrated object measurement, garment measurements, pixel-only measurements when no card is present, and support for tops, pants, shorts, and skirts.

Measurements depend on image quality, segmentation, garment pose, and card placement. Size labels are estimates based on a common shirt chart and the project's reference values; they may differ from a particular brand's sizing.

## Project files

- `main.py` — main garment measurement script. Detects tops, pants, shorts, skirts, and a calibration card. It reports centimeters when a card is found and pixels when no card is present. Pants and shorts include front rise.
- `evaluate_cards.py` — evaluates card detection on a sample card image.
- `measure_any_object.py` — measures a prompted object using a detected card as a scale reference.
- `test_landmarks.py` — extracts clothing landmarks and reports pixel measurements without card calibration.
- `sam3_source/` — SAM 3 source repository, included as a Git submodule.
- `test_cards/` and `test_clothes/` — sample input images.
- `card_results/`, `main_results/`, and `measure_any_object_results/` — saved result overlays and examples. Running the scripts can overwrite the corresponding outputs.

## Clone the project

Clone with the SAM 3 submodule:

```bash
git clone --recurse-submodules https://github.com/ashwinjmonteiro/GarmentSense.git
cd GarmentSense
```

If the repository was cloned without submodules, initialize them with:

```bash
git submodule update --init --recursive
```

## Requirements and setup

- Python 3.10 or newer
- PyTorch and torchvision installed for the local CPU or CUDA setup. Follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).
- Permission to download the SAM 3 checkpoint from Hugging Face. See the [SAM 3 repository](https://github.com/facebookresearch/sam3) for access and authentication instructions. Model weights are not stored in this repository.

From the repository root, install SAM 3 and the image dependencies in an activated virtual environment:

```bash
python -m pip install -e ./sam3_source
python -m pip install opencv-python pillow numpy
```

Install PyTorch before these packages. SAM 3 declares its additional dependencies in `sam3_source/pyproject.toml`.

## Run the scripts

Run commands from the project root so the sample paths resolve:

```bash
python main.py
python evaluate_cards.py
python measure_any_object.py
python test_landmarks.py
```

Each script uses a sample image path in its `__main__` section (or a default command-line path for `test_landmarks.py`). To process another image, change the sample path or call the script's measurement function with the desired path. Results are saved to the output directory shown in the terminal.

## Measurement and validation notes

Card-calibrated dimensions use the standard physical size stored in `main.py`. Card-free runs from `main.py` and `test_landmarks.py` report pixel distances only. Clothing landmarks come from the segmented garment contour, so flat lay, orientation, occlusion, and mask quality affect the estimates. The displayed clothing size is a chart match, not a guarantee of brand fit.

The minimum usable resolution and maximum camera distance have not yet been established under controlled conditions. To determine them, keep the camera, garment, card, background, and lighting fixed; downsample the same image to test resolution; and capture new images at measured distances. Record image dimensions, card and garment pixel sizes, landmark quality, and error against tape-measure dimensions. Repeat captures near the first failures before reporting limits.

## Upstream project

SAM 3 is developed by Meta. See `sam3_source/README.md` and `sam3_source/LICENSE` for upstream documentation and license information. Checkpoint access and use are subject to Meta's terms.
