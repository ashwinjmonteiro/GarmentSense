import os
import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --- Device Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")

if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Lower threshold to reliably catch the card
CONFIDENCE_THRESHOLD = 0.15
print(f"Loading SAM 3 Image Model (Threshold: {CONFIDENCE_THRESHOLD})...")
model = build_sam3_image_model()
model.to(device)
model.eval()

processor = Sam3Processor(model=model, confidence_threshold=CONFIDENCE_THRESHOLD)


def detect_and_mask_card(
    image_path: str,
    output_dir: str = "./card_results",
    prompt: str = "credit card"
):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target image does not exist: {image_path}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # --- 1. Load Image ---
    pil_image = Image.open(image_path).convert("RGB")
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h, w, _ = cv_image.shape

    # --- 2. Run Inference ---
    print(f"\nAnalyzing '{image_path}' with prompt: '{prompt}'...")
    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                inference_state = processor.set_image(pil_image)
                results = processor.set_text_prompt(state=inference_state, prompt=prompt)
        else:
            inference_state = processor.set_image(pil_image)
            results = processor.set_text_prompt(state=inference_state, prompt=prompt)

    # --- 3. Process Predictions ---
    masks = results.get("masks", [])
    scores = results.get("scores", [])

    if torch.is_tensor(scores):
        scores_list = [float(s) for s in scores.flatten().cpu()]
    elif isinstance(scores, list):
        scores_list = [float(s) for s in scores]
    else:
        scores_list = []

    print(f"Confidence scores: {[round(s, 3) for s in scores_list]}")

    num_detected = len(masks) if masks is not None else 0
    if num_detected == 0:
        print(f"No objects detected for '{prompt}'.")
        return False

    # Binary mask canvas
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    overlay = cv_image.copy()

    for idx in range(num_detected):
        mask_np = masks[idx].squeeze().detach().cpu().numpy().astype(bool)
        combined_mask[mask_np] = 255
        score = scores_list[idx] if idx < len(scores_list) else 0.0

        # Create overlay highlight (Cyan: B=255, G=200, R=0)
        colored_mask = np.zeros_like(cv_image, dtype=np.uint8)
        colored_mask[mask_np] = np.array([255, 200, 0], dtype=np.uint8)

        # Alpha blend overlay onto image
        overlay = np.where(mask_np[:, :, None], cv2.addWeighted(overlay, 0.65, colored_mask, 0.35, 0), overlay)

        # Draw green contour around the card boundary
        contours, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        if contours:
            c = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(c)
            label = f"{prompt}: {score:.2f} ({bw}x{bh}px)"
            cv2.putText(overlay, label, (bx, max(by - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Convert single-channel mask to 3-channel BGR so shapes match for horizontal stacking
    mask_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)

    # --- 4. Stitch 3 Panels Together ---
    panel_orig = cv_image.copy()
    panel_overlay = overlay.copy()
    panel_mask = mask_bgr.copy()

    # Draw dark header banner across each tile
    panels = [
        (panel_orig, "Original Image"),
        (panel_overlay, "SAM3 Segmented Overlay"),
        (panel_mask, "Binary Mask")
    ]
    for tile, title in panels:
        cv2.rectangle(tile, (0, 0), (w, 40), (25, 25, 25), -1)
        cv2.putText(tile, title, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Horizontally stitch: [Original | Segmented | Mask]
    stitched_result = np.hstack([panel_orig, panel_overlay, panel_mask])

    # Optional: If the combined image is wider than 1920px, downscale for easy viewing
    stitched_h, stitched_w = stitched_result.shape[:2]
    if stitched_w > 1920:
        scale_ratio = 1920.0 / stitched_w
        stitched_result = cv2.resize(
            stitched_result,
            (1920, int(stitched_h * scale_ratio)),
            interpolation=cv2.INTER_AREA
        )

    # --- 5. Save ONLY the Single Stitched Image ---
    slug = prompt.replace(" ", "_")
    output_path = os.path.join(output_dir, f"{base_name}_{slug}_result.png")
    cv2.imwrite(output_path, stitched_result)

    print(f"Done! Output saved to: {output_path}")
    return True


if __name__ == "__main__":
    TEST_IMAGE = "test_cards/card9.jpg"

    success = detect_and_mask_card(
        image_path=TEST_IMAGE,
        output_dir="./card_results",
        prompt="credit card"
    )

    if not success:
        detect_and_mask_card(
            image_path=TEST_IMAGE,
            output_dir="./card_results",
            prompt="credit card"
        )