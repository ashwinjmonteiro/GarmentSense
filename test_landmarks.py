"""Measure tops, pants, or skirts in pixels without card calibration.

Landmark extraction and garment classification are imported directly from
main.py. This script reports the same garment measurements in raw pixels.

Examples:
    python sub_prompt.py test_clothes/cloth01.jpg
    python sub_prompt.py test_clothes/long_pants.jpg --output-dir test_measurement_results
"""

import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image

from main import (
    classify_garment_with_variants,
    device,
    extract_pants_landmarks,
    extract_skirt_landmarks,
    extract_top_landmarks,
    processor,
)


DEFAULT_IMAGE = "test_clothes/001240.jpg"
DEFAULT_OUTPUT_DIR = "test_measurement_results"


def distance_px(point_a, point_b):
    """Euclidean distance between two image coordinates, in pixels."""
    if point_a is None or point_b is None:
        return 0.0
    return float(np.linalg.norm(np.asarray(point_a) - np.asarray(point_b)))


def draw_measurement(image, point_a, point_b, text, color, offset=(0, -10)):
    """Draw the same landmark pair used by main.py."""
    if point_a is None or point_b is None:
        return

    point_a = tuple(map(int, point_a))
    point_b = tuple(map(int, point_b))
    cv2.line(image, point_a, point_b, color, 3, cv2.LINE_AA)
    for point in (point_a, point_b):
        cv2.circle(image, point, 6, (0, 0, 255), -1)
        cv2.circle(image, point, 7, (0, 0, 0), 1, cv2.LINE_AA)

    mid_x = (point_a[0] + point_b[0]) // 2 + offset[0]
    mid_y = (point_a[1] + point_b[1]) // 2 + offset[1]
    label = f"{text}: {distance_px(point_a, point_b):.1f} px"
    text_x = max(4, min(mid_x, image.shape[1] - 4))
    text_y = max(18, min(mid_y, image.shape[0] - 4))
    cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)


def measure_garment_pixels(image_path, output_dir=DEFAULT_OUTPUT_DIR):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    os.makedirs(output_dir, exist_ok=True)
    pil_image = Image.open(image_path).convert("RGB")
    image = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]

    # Use the same SAM 3 garment classification as geometric_contour.py.
    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = processor.set_image(pil_image)
                garment_type, garment_mask, detected_label = classify_garment_with_variants(
                    processor, state, (height, width)
                )
        else:
            state = processor.set_image(pil_image)
            garment_type, garment_mask, detected_label = classify_garment_with_variants(
                processor, state, (height, width)
            )

    if garment_mask is None:
        raise RuntimeError("Could not segment a shirt, t-shirt, pair of pants, shorts, or skirt in this image.")

    # Keep main.py's existing top and pants landmark logic unchanged.
    if garment_type in {"pants", "shorts"}:
        landmarks = extract_pants_landmarks(garment_mask)
    elif garment_type == "skirt":
        landmarks = extract_skirt_landmarks(garment_mask)
    else:
        landmarks = extract_top_landmarks(garment_mask)
    if landmarks is None:
        raise RuntimeError("Could not extract landmarks from the segmented garment.")

    output = image.copy()
    overlay_color = np.zeros_like(image)
    overlay_color[:] = (0, 180, 0)
    output[garment_mask] = cv2.addWeighted(image, 0.70, overlay_color, 0.30, 0)[garment_mask]
    contours, _ = cv2.findContours(
        garment_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

    print(f"\nDetected category: {garment_type.upper()} ({detected_label})")
    print("Measurements are pixel distances in the original image; no card is used.")

    if garment_type in {"pants", "shorts"}:
        measurements = [
            ("Waist (half width)", "waist_left", "waist_right", (0, 255, 255), (-60, -15)),
            ("Hip width", "hip_left", "hip_right", (255, 100, 255), (-50, -10)),
            ("Outseam", "waist_left", "left_cuff_out", (0, 220, 0), (-150, 0)),
            ("Inseam", "crotch", "left_cuff_center", (0, 165, 255), (10, 0)),
            ("Cuff width", "left_cuff_out", "left_cuff_in", (255, 50, 50), (-40, 25)),
        ]
    elif garment_type == "skirt":
        measurements = [
            ("Waist width", "waist_left", "waist_right", (0, 255, 255), (0, -15)),
            ("Hip width", "hip_left", "hip_right", (255, 100, 255), (0, -10)),
            ("Hem width", "hem_left", "hem_right", (255, 50, 50), (0, 25)),
            ("Skirt length", "top_center", "hem_center", (0, 220, 0), (10, 0)),
        ]
    else:
        measurements = [
            ("Shoulders", "left_shoulder", "right_shoulder", (200, 50, 180), (-110, 28)),
            ("Chest width", "left_armpit", "right_armpit", (230, 160, 40), (-80, -12)),
            ("Body length", "collar", "hem", (0, 220, 0), (-190, 80)),
            ("Left sleeve", "left_shoulder", "left_cuff", (255, 80, 80), (-80, -15)),
            ("Right sleeve", "right_shoulder", "right_cuff", (255, 80, 80), (10, -15)),
        ]

    for label, start_key, end_key, color, offset in measurements:
        start, end = landmarks[start_key], landmarks[end_key]
        value = distance_px(start, end)
        print(f"{label}: {value:.1f} px")
        draw_measurement(output, start, end, label, color, offset)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_pixel_measurements.jpg")
    if not cv2.imwrite(output_path, output):
        raise OSError(f"Could not save result image: {output_path}")
    print(f"Saved result: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure tops, pants, or skirts in pixels without card calibration."
    )
    parser.add_argument("image", nargs="?", default=DEFAULT_IMAGE,
                        help=f"Image to process (default: {DEFAULT_IMAGE})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Result directory (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()
    measure_garment_pixels(args.image, args.output_dir)


if __name__ == "__main__":
    main()
