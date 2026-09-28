import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

CARD_REAL_WIDTH_CM = 8.56
CARD_REAL_HEIGHT_CM = 5.398
CM_PER_INCH = 2.54

TOP_SIZE_REFERENCE = [
    ("S", 40.0, 17.0, 28.5),
    ("M", 42.0, 17.5, 29.0),
    ("L", 44.0, 18.0, 29.5),
    ("XL", 46.0, 18.5, 30.0),
    ("XXL", 48.0, 19.0, 30.5),
]

PANTS_SIZE_REFERENCE = [
    ("28 / XS", 28.0, 37.0),
    ("30 / S", 30.0, 39.0),
    ("32 / M", 32.0, 41.0),
    ("34 / L", 34.0, 43.0),
    ("36 / XL", 36.0, 45.0),
    ("38 / XXL", 38.0, 47.0),
]

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


def euclidean_dist_cm(p1, p2, ppm):
    if p1 is None or p2 is None or ppm <= 0:
        return 0.0
    dist_px = np.linalg.norm(np.array(p1) - np.array(p2))
    return float(dist_px / ppm)


def closest_top_size(chest_width_cm, shoulder_cm, body_length_cm):
    measured = np.array(
        [chest_width_cm * 2.0, shoulder_cm, body_length_cm], dtype=float
    )
    scales = np.array([2.0, 0.5, 0.5], dtype=float) * CM_PER_INCH
    weights = np.array([2.0, 2.0, 1.0], dtype=float)

    best_size = None
    best_score = float("inf")
    for label, chest_in, shoulder_in, length_in in TOP_SIZE_REFERENCE:
        reference = np.array([chest_in, shoulder_in, length_in]) * CM_PER_INCH
        score = np.sum(weights * ((measured - reference) / scales) ** 2)
        if score < best_score:
            best_size, best_score = label, score
    return best_size


def closest_pants_size(waist_width_cm, hip_width_cm):
    measured = np.array([waist_width_cm * 2.0, hip_width_cm * 2.0])
    scales = np.array([2.0, 2.0]) * CM_PER_INCH
    best_size = None
    best_score = float("inf")

    for label, waist_in, hip_in in PANTS_SIZE_REFERENCE:
        reference = np.array([waist_in, hip_in]) * CM_PER_INCH
        score = np.sum(((measured - reference) / scales) ** 2)
        if score < best_score:
            best_size, best_score = label, score
    return best_size


def draw_size_label(image, size_text):
    text = f"Estimated India men's size: {size_text}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = 24, 48
    cv2.rectangle(
        image,
        (x - 10, y - text_h - 12),
        (x + text_w + 10, y + baseline + 8),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        image, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA
    )


def classify_garment(processor, state, image_shape):
    """
    Evaluates both 'shirt' and 'pants' prompts with SAM 3, then verifies
    geometrically via morphological leg bifurcation detection.
    """
    top_prompts = ["shirt", "t-shirt"]
    bottom_prompts = ["pants", "trousers"]

    best_top_mask, best_top_score = None, -1.0
    best_top_name = "shirt"
    for p in top_prompts:
        m, _, s = get_mask_and_box(processor, state, p, image_shape)
        if s > best_top_score:
            best_top_score = s
            best_top_mask = m
            best_top_name = p

    best_bot_mask, best_bot_score = None, -1.0
    best_bot_name = "pants"
    for p in bottom_prompts:
        m, _, s = get_mask_and_box(processor, state, p, image_shape)
        if s > best_bot_score:
            best_bot_score = s
            best_bot_mask = m
            best_bot_name = p

    if best_bot_score > best_top_score:
        garment_type = "pants"
        active_mask = best_bot_mask
        label = best_bot_name
    else:
        garment_type = "top"
        active_mask = best_top_mask if best_top_mask is not None else best_bot_mask
        label = best_top_name

    if active_mask is not None:
        mask_u8 = (active_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            y, h = cv2.boundingRect(cnt)[1], cv2.boundingRect(cnt)[3]
            slice_y = int(y + h * 0.85)
            if slice_y < mask_u8.shape[0]:
                row = mask_u8[slice_y, :]
                diff = np.diff(row.astype(int))
                transitions = np.count_nonzero(diff != 0)
                if transitions >= 4:
                    garment_type = "pants"
                    label = "pants"

    return garment_type, active_mask, label


def extract_top_landmarks(clothing_mask):
    mask_u8 = (clothing_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    pts = contour.reshape(-1, 2)

    corridor_half_w = max(20, int(clothing_mask.shape[1] * 0.03))
    central_pts = pts[np.abs(pts[:, 0] - cx) <= corridor_half_w]
    collar_pt = tuple(central_pts[np.argmin(central_pts[:, 1])]) if len(central_pts) > 0 else (cx, int(np.min(pts[:, 1])))
    hem_pt = tuple(central_pts[np.argmax(central_pts[:, 1])]) if len(central_pts) > 0 else (cx, int(np.max(pts[:, 1])))
    total_h = max(10, hem_pt[1] - collar_pt[1])

    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)

    left_armpits, right_armpits = [], []
    if defects is not None:
        for i in range(defects.shape[0]):
            defect = np.asarray(defects[i]).reshape(-1)

            if defect.size != 4:
                continue

            s, e, f, d = defect
            depth = d / 256.0
            pt = tuple(contour[f][0])
            if depth > 18.0 and collar_pt[1] + 25 < pt[1] < hem_pt[1] - 30:
                if pt[0] < cx:
                    left_armpits.append((depth, pt))
                else:
                    right_armpits.append((depth, pt))

    is_flat_lay = len(left_armpits) > 0 and len(right_armpits) > 0
    if is_flat_lay:
        left_pit_y = max(left_armpits, key=lambda item: item[0])[1][1]
        right_pit_y = max(right_armpits, key=lambda item: item[0])[1][1]
        is_flat_lay = (
            (left_pit_y + right_pit_y) / 2
            < collar_pt[1] + total_h * 0.45
        )

    if is_flat_lay:
            left_armpit = max(left_armpits, key=lambda x: x[0])[1]
            right_armpit = max(right_armpits, key=lambda x: x[0])[1]

            def get_flat_shoulder(neck_pt, pit_pt, is_left=True):
                if is_left:
                    min_x = int(neck_pt[0] - abs(neck_pt[0] - pit_pt[0]) * 0.95)
                    max_x = int(neck_pt[0] - abs(neck_pt[0] - pit_pt[0]) * 0.40)
                else:
                    min_x = int(neck_pt[0] + abs(neck_pt[0] - pit_pt[0]) * 0.40)
                    max_x = int(neck_pt[0] + abs(neck_pt[0] - pit_pt[0]) * 0.95)

                zone = pts[
                    (pts[:, 0] >= min_x) &
                    (pts[:, 0] <= max_x) &
                    (pts[:, 1] <= pit_pt[1])
                ]

                if len(zone) == 0:
                    return (
                        int((neck_pt[0] + pit_pt[0]) / 2),
                        neck_pt[1] + 25
                    )

                unique_xs = np.unique(zone[:, 0])
                top_edge = np.array([
                    [ux, np.min(zone[zone[:, 0] == ux][:, 1])]
                    for ux in unique_xs
                ])

                if len(top_edge) < 5:
                    return tuple(top_edge[len(top_edge) // 2])


                smooth_window = 9

                if len(top_edge) >= smooth_window:
                    kernel = np.ones(smooth_window) / smooth_window

                    smooth_x = np.convolve(
                        top_edge[:, 0],
                        kernel,
                        mode="valid"
                    )

                    smooth_y = np.convolve(
                        top_edge[:, 1],
                        kernel,
                        mode="valid"
                    )

                    smooth_edge = np.column_stack((smooth_x, smooth_y))
                else:
                    smooth_edge = top_edge.astype(float)

                if len(smooth_edge) < 5:
                    return tuple(top_edge[len(top_edge) // 2])

                angles = []

                for i in range(2, len(smooth_edge) - 2):
                    v1 = smooth_edge[i] - smooth_edge[i - 2]
                    v2 = smooth_edge[i + 2] - smooth_edge[i]

                    n1 = np.linalg.norm(v1)
                    n2 = np.linalg.norm(v2)

                    if n1 == 0 or n2 == 0:
                        angles.append(0.0)
                        continue

                    cosine = np.dot(v1, v2) / (n1 * n2)
                    cosine = np.clip(cosine, -1.0, 1.0)

                    angle = np.arccos(cosine)
                    angles.append(angle)

                if not angles:
                    return tuple(top_edge[len(top_edge) // 2])

                margin = max(2, int(len(angles) * 0.08))

                search_start = margin
                search_end = len(angles) - margin

                if search_end <= search_start:
                    return tuple(top_edge[len(top_edge) // 2])

                local_angles = np.array(angles[search_start:search_end])

                best_local_idx = int(np.argmax(local_angles))
                best_idx = best_local_idx + search_start + 2

                shoulder_pt = smooth_edge[best_idx]

                return (
                    int(round(shoulder_pt[0])),
                    int(round(shoulder_pt[1]))
                )

            left_shoulder = get_flat_shoulder(collar_pt, left_armpit, is_left=True)
            right_shoulder = get_flat_shoulder(collar_pt, right_armpit, is_left=False)

            left_cuff = tuple(pts[np.argmin(pts[:, 0])])
            right_cuff = tuple(pts[np.argmax(pts[:, 0])])
    else:
        collar_pt = (
            tuple(central_pts[np.argmin(central_pts[:, 1])])
            if len(central_pts) > 0
            else (cx, int(np.min(pts[:, 1])))
        )
        hem_pt = (
            tuple(central_pts[np.argmax(central_pts[:, 1])])
            if len(central_pts) > 0
            else (cx, int(np.max(pts[:, 1])))
        )
        total_h = max(10, hem_pt[1] - collar_pt[1])

        y_sh_top = int(collar_pt[1] + total_h * 0.08)
        y_sh_bot = int(collar_pt[1] + total_h * 0.18)
        shoulder_pts = pts[
            (pts[:, 1] >= y_sh_top) & (pts[:, 1] <= y_sh_bot)
        ]
        left_cands = shoulder_pts[shoulder_pts[:, 0] < cx]
        right_cands = shoulder_pts[shoulder_pts[:, 0] > cx]

        left_shoulder = (
            tuple(left_cands[np.argmin(left_cands[:, 0])])
            if len(left_cands) > 0
            else (int(cx - total_h * 0.3), y_sh_top + 15)
        )
        right_shoulder = (
            tuple(right_cands[np.argmax(right_cands[:, 0])])
            if len(right_cands) > 0
            else (int(cx + total_h * 0.3), y_sh_top + 15)
        )

        y_chest = int(collar_pt[1] + total_h * 0.39)
        y_trunk = int(collar_pt[1] + total_h * 0.65)
        trunk_band = pts[np.abs(pts[:, 1] - y_trunk) <= 15]
        left_trunk = trunk_band[trunk_band[:, 0] < cx]
        right_trunk = trunk_band[trunk_band[:, 0] > cx]

        if len(left_trunk) > 0 and len(right_trunk) > 0:
            trunk_half_w = (
                np.max(right_trunk[:, 0]) - np.min(left_trunk[:, 0])
            ) / 2.0
            chest_half_w = trunk_half_w * 1.07
            left_armpit = (int(cx - chest_half_w), y_chest)
            right_armpit = (int(cx + chest_half_w), y_chest)
        else:
            shoulder_half_w = abs(right_shoulder[0] - left_shoulder[0]) / 2.0
            left_armpit = (int(cx - shoulder_half_w * 1.05), y_chest)
            right_armpit = (int(cx + shoulder_half_w * 1.05), y_chest)

        y_cuff_min = int(collar_pt[1] + total_h * 0.40)
        y_cuff_max = int(collar_pt[1] + total_h * 0.60)
        cuff_band = pts[
            (pts[:, 1] >= y_cuff_min) & (pts[:, 1] <= y_cuff_max)
        ]
        left_cuffs = cuff_band[cuff_band[:, 0] < cx]
        right_cuffs = cuff_band[cuff_band[:, 0] > cx]

        left_cuff = (
            tuple(left_cuffs[np.argmin(left_cuffs[:, 0])])
            if len(left_cuffs) > 0
            else tuple(pts[np.argmin(pts[:, 0])])
        )
        right_cuff = (
            tuple(right_cuffs[np.argmax(right_cuffs[:, 0])])
            if len(right_cuffs) > 0
            else tuple(pts[np.argmax(pts[:, 0])])
        )

    return {
        "collar": collar_pt,
        "hem": hem_pt,
        "left_armpit": left_armpit,
        "right_armpit": right_armpit,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "left_cuff": left_cuff,
        "right_cuff": right_cuff
    }


def extract_pants_landmarks(pants_mask):
    mask_u8 = (pants_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    pts = contour.reshape(-1, 2)

    y_min = np.min(pts[:, 1])
    waist_band = pts[pts[:, 1] <= y_min + 20]
    waist_left = tuple(waist_band[np.argmin(waist_band[:, 0])])
    waist_right = tuple(waist_band[np.argmax(waist_band[:, 0])])

    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)

    crotch_pt = None
    if defects is not None:
        crotch_cands = []
        for i in range(defects.shape[0]):
            defect = np.asarray(defects[i]).reshape(-1)

            if defect.size != 4:
                continue

            s, e, f, d = defect
            depth = d / 256.0
            pt = tuple(contour[f][0])
            if depth > 20.0 and abs(pt[0] - cx) < pants_mask.shape[1] * 0.18 and pt[1] > y_min + 50:
                crotch_cands.append((depth, pt))

        if crotch_cands:
            crotch_pt = max(crotch_cands, key=lambda x: x[0])[1]

    if crotch_pt is None:
        crotch_pt = (cx, int(y_min + (np.max(pts[:, 1]) - y_min) * 0.35))

    total_h = np.max(pts[:, 1]) - y_min

    hip_zone = pts[(pts[:, 1] > y_min + 15) & (pts[:, 1] < crotch_pt[1])]
    if len(hip_zone) > 0:
        hip_left_cand = hip_zone[np.argmin(hip_zone[:, 0])]
        hip_right_cand = hip_zone[np.argmax(hip_zone[:, 0])]
        y_hip = int((hip_left_cand[1] + hip_right_cand[1]) / 2)
        hip_left = (int(hip_left_cand[0]), y_hip)
        hip_right = (int(hip_right_cand[0]), y_hip)
    else:
        y_hip = int(y_min + total_h * 0.20)
        hip_left = (waist_left[0] - 10, y_hip)
        hip_right = (waist_right[0] + 10, y_hip)

    left_leg_all = pts[pts[:, 0] < cx]
    if len(left_leg_all) > 0:
        y_left_max = np.max(left_leg_all[:, 1])
        left_hem_pts = left_leg_all[left_leg_all[:, 1] >= y_left_max - 25]

        left_cuff_out = tuple(left_hem_pts[np.argmin(left_hem_pts[:, 0])])
        left_cuff_in = tuple(left_hem_pts[np.argmax(left_hem_pts[:, 0])])
        left_cuff_center = (int((left_cuff_out[0] + left_cuff_in[0]) / 2),
                            int((left_cuff_out[1] + left_cuff_in[1]) / 2))
    else:
        left_cuff_out = (int(cx * 0.5), int(np.max(pts[:, 1])))
        left_cuff_in = (int(cx * 0.8), int(np.max(pts[:, 1])))
        left_cuff_center = (int((left_cuff_out[0] + left_cuff_in[0]) / 2), left_cuff_out[1])

    right_leg_all = pts[pts[:, 0] > cx]
    if len(right_leg_all) > 0:
        y_right_max = np.max(right_leg_all[:, 1])
        right_hem_pts = right_leg_all[right_leg_all[:, 1] >= y_right_max - 25]
        right_cuff_in = tuple(right_hem_pts[np.argmin(right_hem_pts[:, 0])])
        right_cuff_out = tuple(right_hem_pts[np.argmax(right_hem_pts[:, 0])])
        right_cuff_center = (int((right_cuff_out[0] + right_cuff_in[0]) / 2),
                             int((right_cuff_out[1] + right_cuff_in[1]) / 2))
    else:
        right_cuff_in = (int(cx * 1.2), int(np.max(pts[:, 1])))
        right_cuff_out = (int(cx * 1.5), int(np.max(pts[:, 1])))
        right_cuff_center = (int((right_cuff_out[0] + right_cuff_in[0]) / 2), right_cuff_out[1])

    return {
        "waist_left": waist_left,
        "waist_right": waist_right,
        "hip_left": hip_left,
        "hip_right": hip_right,
        "crotch": crotch_pt,
        "left_cuff_out": left_cuff_out,
        "left_cuff_in": left_cuff_in,
        "left_cuff_center": left_cuff_center,
        "right_cuff_out": right_cuff_out,
        "right_cuff_in": right_cuff_in,
        "right_cuff_center": right_cuff_center
    }


def draw_styled_measurement(img, p1, p2, text, line_color, text_color=(255, 255, 255), offset=(0, -10)):
    if p1 is None or p2 is None:
        return
    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))

    cv2.line(img, p1, p2, line_color, 3, cv2.LINE_AA)
    cv2.circle(img, p1, 6, (0, 0, 255), -1)
    cv2.circle(img, p2, 6, (0, 0, 255), -1)
    cv2.circle(img, p1, 7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(img, p2, 7, (0, 0, 0), 1, cv2.LINE_AA)

    mx = int((p1[0] + p2[0]) / 2) + offset[0]
    my = int((p1[1] + p2[1]) / 2) + offset[1]

    cv2.putText(img, text, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2, cv2.LINE_AA)


def measure_garment_auto(
    image_path: str,
    card_prompt: str = "credit card",
    output_dir: str = "./contour_measurement_results"
):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target image not found: {image_path}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    pil_image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h, w, _ = cv_image.shape

    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = processor.set_image(pil_image)
                card_mask, card_rect, _ = get_mask_and_box(processor, state, card_prompt, (h, w))
                if card_mask is None and card_prompt != "card":
                    card_mask, card_rect, _ = get_mask_and_box(processor, state, "card", (h, w))
                garment_type, cloth_mask, detected_label = classify_garment(processor, state, (h, w))
        else:
            state = processor.set_image(pil_image)
            card_mask, card_rect, _ = get_mask_and_box(processor, state, card_prompt, (h, w))
            garment_type, cloth_mask, detected_label = classify_garment(processor, state, (h, w))

    if card_mask is None:
        print("Error: Could not detect calibration card.")
        return

    if cloth_mask is None:
        print("Error: Could not segment clothing item.")
        return

    (_, _), (dim_a, dim_b), _ = card_rect
    ppm = (max(dim_a, dim_b) / CARD_REAL_WIDTH_CM + min(dim_a, dim_b) / CARD_REAL_HEIGHT_CM) / 2.0

    output_img = cv_image.copy()
    print(f"\n[Detected Category]: {garment_type.upper()} ({detected_label})")
    print(f"[Calibration PPM]:   {ppm:.2f} px/cm")

    if garment_type == "pants":
        lm = extract_pants_landmarks(cloth_mask)
        if lm is None:
            return

        waist_cm = euclidean_dist_cm(lm["waist_left"], lm["waist_right"], ppm)
        hip_cm = euclidean_dist_cm(lm["hip_left"], lm["hip_right"], ppm)
        outseam_cm = euclidean_dist_cm(lm["waist_left"], lm["left_cuff_out"], ppm)
        inseam_cm = euclidean_dist_cm(lm["crotch"], lm["left_cuff_center"], ppm)
        cuff_cm = euclidean_dist_cm(lm["left_cuff_out"], lm["left_cuff_in"], ppm)

        print("\n--- Pants Measurements ---")
        print(f"Waist Width:        {waist_cm:.1f} cm (Half) | {waist_cm*2:.1f} cm (Full)")
        print(f"Hip Width:          {hip_cm:.1f} cm")
        print(f"Outseam (Total L):  {outseam_cm:.1f} cm")
        print(f"Inseam:             {inseam_cm:.1f} cm")
        print(f"Cuff/Hem Width:     {cuff_cm:.1f} cm")
        size_label = closest_pants_size(waist_cm, hip_cm)
        print(f"Estimated India men's size: {size_label} (reference-chart match)")

        draw_styled_measurement(output_img, lm["waist_left"], lm["waist_right"], f"Waist: {waist_cm:.1f} cm", (0, 255, 255), offset=(-60, -15))
        draw_styled_measurement(output_img, lm["hip_left"], lm["hip_right"], f"Hips: {hip_cm:.1f} cm", (255, 100, 255), offset=(-50, -10))
        draw_styled_measurement(output_img, lm["waist_left"], lm["left_cuff_out"], f"Outseam: {outseam_cm:.1f} cm", (0, 220, 0), offset=(-150, 0))
        draw_styled_measurement(output_img, lm["crotch"], lm["left_cuff_center"], f"Inseam: {inseam_cm:.1f} cm", (0, 165, 255), offset=(10, 0))
        draw_styled_measurement(output_img, lm["left_cuff_out"], lm["left_cuff_in"], f"Cuff: {cuff_cm:.1f} cm", (255, 50, 50), offset=(-40, 25))

    else:
        lm = extract_top_landmarks(cloth_mask)
        if lm is None:
            return

        shoulders_cm = euclidean_dist_cm(lm["left_shoulder"], lm["right_shoulder"], ppm)
        chest_cm = euclidean_dist_cm(lm["left_armpit"], lm["right_armpit"], ppm)
        body_length_cm = euclidean_dist_cm(lm["collar"], lm["hem"], ppm)
        left_sleeve_cm = euclidean_dist_cm(lm["left_shoulder"], lm["left_cuff"], ppm)
        right_sleeve_cm = euclidean_dist_cm(lm["right_shoulder"], lm["right_cuff"], ppm)

        print("\n--- Top Measurements ---")
        print(f"Shoulders:    {shoulders_cm:.1f} cm")
        print(f"Chest Width:  {chest_cm:.1f} cm")
        print(f"Body Length:  {body_length_cm:.1f} cm")
        print(f"Left Sleeve:  {left_sleeve_cm:.1f} cm")
        print(f"Right Sleeve: {right_sleeve_cm:.1f} cm")
        size_label = closest_top_size(chest_cm, shoulders_cm, body_length_cm)
        print(f"Estimated India men's size: {size_label} (reference-chart match)")

        draw_styled_measurement(output_img, lm["left_shoulder"], lm["right_shoulder"], f"Shoulders: {shoulders_cm:.1f} cm", (200, 50, 180), offset=(-110, 28))
        draw_styled_measurement(output_img, lm["left_armpit"], lm["right_armpit"], f"Chest: {chest_cm:.1f} cm", (230, 160, 40), offset=(-80, -12))
        draw_styled_measurement(output_img, lm["collar"], lm["hem"], f"Body Length: {body_length_cm:.1f} cm", (0, 220, 0), offset=(-190, 80))
        draw_styled_measurement(output_img, lm["left_shoulder"], lm["left_cuff"], f"Sleeve L: {left_sleeve_cm:.1f} cm", (255, 80, 80), offset=(-80, -15))
        draw_styled_measurement(output_img, lm["right_shoulder"], lm["right_cuff"], f"Sleeve R: {right_sleeve_cm:.1f} cm", (255, 80, 80), offset=(10, -15))

    draw_size_label(output_img, size_label)
    out_file = os.path.join(output_dir, f"{base_name}_measurements.png")
    cv2.imwrite(out_file, output_img)
    print(f"\nResult saved to: {out_file}")


if __name__ == "__main__":
    measure_garment_auto("../images/t-shirt/t1.png")
