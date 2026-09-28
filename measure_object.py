import os
import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --- Standard Card Physical Dimensions (in cm) ---
CARD_REAL_WIDTH_CM = 8.56    # 85.60 mm
CARD_REAL_HEIGHT_CM = 5.398  # 53.98 mm

# --- Device Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")

if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print("Loading SAM 3 Image Model...")
model = build_sam3_image_model()
model.to(device)
model.eval()

processor = Sam3Processor(model=model, confidence_threshold=0.15)


def get_mask_and_box(processor, inference_state, prompt: str, image_shape):
    """Queries SAM 3 and returns the mask, minimum bounding box, and score."""
    h, w = image_shape[:2]
    results = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks = results.get("masks", [])
    scores = results.get("scores", [])

    if masks is None or len(masks) == 0:
        return None, None, 0.0

    if torch.is_tensor(scores):
        scores_list = [float(s) for s in scores.flatten().cpu()]
    elif isinstance(scores, list):
        scores_list = [float(s) for s in scores]
    else:
        scores_list = [0.0] * len(masks)

    best_idx = int(np.argmax(scores_list))
    best_mask = masks[best_idx].squeeze().detach().cpu().numpy().astype(bool)
    best_score = scores_list[best_idx]

    contours, _ = cv2.findContours(best_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0.0

    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)

    return best_mask, rect, best_score


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def draw_dimensioned_box(image, box_pts, dim1_cm, dim2_cm, color=(0, 255, 0), label_prefix=""):
    """
    Draws an oriented bounding box with measurement lines,
    ticks, and centered dimension labels along the edges.
    """
    pts = np.int32(box_pts)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)

    # Edge midpoints
    tl, tr, br, bl = box_pts[0], box_pts[1], box_pts[2], box_pts[3]
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # Draw edge-to-edge guideline dashes
    cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), color, 1, cv2.LINE_AA)
    cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), color, 1, cv2.LINE_AA)

    # Draw small endpoint circles
    for pt in [tl, tr, br, bl]:
        cv2.circle(image, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)

    # Calculate actual lengths of adjacent edges
    d_A = np.linalg.norm(tl - tr)
    d_B = np.linalg.norm(tr - br)

    # Match dimensions so the longer real cm maps to the longer pixel edge
    if d_A >= d_B:
        label_A = f"{max(dim1_cm, dim2_cm):.2f} cm"
        label_B = f"{min(dim1_cm, dim2_cm):.2f} cm"
    else:
        label_A = f"{min(dim1_cm, dim2_cm):.2f} cm"
        label_B = f"{max(dim1_cm, dim2_cm):.2f} cm"

    # Edge 1 Label (tl -> tr)
    cv2.putText(
        image, label_A, (int(tltrX - 35), int(tltrY - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA
    )
    cv2.putText(
        image, label_A, (int(tltrX - 35), int(tltrY - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA
    )

    # Edge 2 Label (tr -> br)
    cv2.putText(
        image, label_B, (int(trbrX + 10), int(trbrY)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA
    )
    cv2.putText(
        image, label_B, (int(trbrX + 10), int(trbrY)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA
    )

    # Main object header tag
    if label_prefix:
        center_x = int((tl[0] + br[0]) / 2)
        center_y = int((tl[1] + br[1]) / 2)
        cv2.putText(
            image, label_prefix, (center_x - 45, center_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            image, label_prefix, (center_x - 45, center_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA
        )


def measure_objects_with_card(
    image_path: str,
    target_prompt: str,
    card_prompt: str = "credit card",
    output_dir: str = "./measurement_results"
):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target image not found: {image_path}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    pil_image = Image.open(image_path).convert("RGB")
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h, w, _ = cv_image.shape

    # --- Run SAM 3 Inference ---
    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                inference_state = processor.set_image(pil_image)
                card_mask, card_rect, _ = get_mask_and_box(processor, inference_state, card_prompt, (h, w))
                if card_mask is None and card_prompt != "card":
                    card_mask, card_rect, _ = get_mask_and_box(processor, inference_state, "card", (h, w))
                obj_mask, obj_rect, _ = get_mask_and_box(processor, inference_state, target_prompt, (h, w))
        else:
            inference_state = processor.set_image(pil_image)
            card_mask, card_rect, _ = get_mask_and_box(processor, inference_state, card_prompt, (h, w))
            obj_mask, obj_rect, _ = get_mask_and_box(processor, inference_state, target_prompt, (h, w))

    if card_mask is None:
        print("Error: Could not segment calibration card.")
        return False

    if obj_mask is None:
        print(f"Error: Could not segment target object '{target_prompt}'.")
        return False

    # --- Calibration Factor ---
    (_, _), (dim_a, dim_b), _ = card_rect
    card_w_px = max(dim_a, dim_b)
    card_h_px = min(dim_a, dim_b)

    ppm_w = card_w_px / CARD_REAL_WIDTH_CM
    ppm_h = card_h_px / CARD_REAL_HEIGHT_CM
    pixels_per_cm = (ppm_w + ppm_h) / 2.0

    # --- Object Measurement in CM ---
    (_, _), (odim_a, odim_b), _ = obj_rect
    obj_len_px = max(odim_a, odim_b)
    obj_wid_px = min(odim_a, odim_b)

    obj_length_cm = obj_len_px / pixels_per_cm
    obj_width_cm = obj_wid_px / pixels_per_cm

    print(f"\n[Target: '{target_prompt}']")
    print(f"  Dimensions: {obj_length_cm:.2f} cm x {obj_width_cm:.2f} cm")
    print(f"  PPM Ratio:  {pixels_per_cm:.2f} px/cm\n")

    # --- Drawing Overlays ---
    overlay = cv_image.copy()

    # Semi-transparent color masks
    overlay[card_mask] = cv2.addWeighted(overlay, 0.65, np.full_like(cv_image, (0, 165, 255)), 0.35, 0)[card_mask]
    overlay[obj_mask] = cv2.addWeighted(overlay, 0.65, np.full_like(cv_image, (255, 200, 0)), 0.35, 0)[obj_mask]

    # Get ordered bounding box corners
    card_box = cv2.boxPoints(card_rect)
    obj_box = cv2.boxPoints(obj_rect)

    # Draw measurements on the card (Reference: 8.56 x 5.40 cm)
    draw_dimensioned_box(
        overlay, card_box,
        CARD_REAL_WIDTH_CM, CARD_REAL_HEIGHT_CM,
        color=(0, 215, 255), label_prefix="CARD [REF]"
    )

    # Draw calculated measurements on the target object
    draw_dimensioned_box(
        overlay, obj_box,
        obj_length_cm, obj_width_cm,
        color=(0, 255, 0), label_prefix=target_prompt.upper()
    )

    # --- Combined Mask Panel ---
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    combined_mask[card_mask] = 180
    combined_mask[obj_mask] = 255
    mask_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)

    # --- Horizontal Stitched Output ---
    panel_orig = cv_image.copy()
    panel_overlay = overlay.copy()
    panel_mask = mask_bgr.copy()

    tiles = [
        (panel_orig, "Original Image"),
        (panel_overlay, f"{target_prompt.upper()}: {obj_length_cm:.2f}cm x {obj_width_cm:.2f}cm"),
        (panel_mask, "Binary Masks")
    ]
    for tile, title in tiles:
        cv2.rectangle(tile, (0, 0), (w, 40), (25, 25, 25), -1)
        cv2.putText(tile, title, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    stitched = np.hstack([panel_orig, panel_overlay, panel_mask])

    sh, sw = stitched.shape[:2]
    if sw > 1920:
        stitched = cv2.resize(stitched, (1920, int(sh * (1920.0 / sw))), interpolation=cv2.INTER_AREA)

    output_path = os.path.join(output_dir, f"{base_name}_{target_prompt}_measured.png")
    cv2.imwrite(output_path, stitched)
    print(f"Result saved to: {output_path}")
    return True


if __name__ == "__main__":
    TEST_IMAGE = "test_cards/card18.jpeg"

    # Set prompt to whatever object is placed next to your card
    TARGET_OBJECT_PROMPT = "bottle"

    measure_objects_with_card(
        image_path=TEST_IMAGE,
        target_prompt=TARGET_OBJECT_PROMPT,
        card_prompt="credit card"
    )