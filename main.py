import os
import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

CARD_REAL_WIDTH_CM = 8.56
CARD_REAL_HEIGHT_CM = 5.398
CM_PER_INCH = 2.54

TOP_SIZE_REFERENCE = [
    # Common Indian shirt garment measurements in inches:
    # label, chest circumference, shoulder width, body length, long sleeve.
    ("S", 38.3, 16.0, 28.3, 22.0),
    ("M", 40.5, 17.0, 29.0, 23.0),
    ("L", 43.8, 18.0, 29.8, 24.0),
    ("XL", 46.0, 19.0, 30.5, 26.0),
    ("XXL", 49.3, 20.0, 31.3, 28.0),
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


def closest_top_size(
    chest_width_cm,
    shoulder_cm,
    body_length_cm,
    left_sleeve_cm=None,
    right_sleeve_cm=None,
):
    """Match tops against chest, shoulder, and length; include full sleeves when present."""
    labels = [row[0] for row in TOP_SIZE_REFERENCE]
    chest_in = chest_width_cm * 2.0 / CM_PER_INCH
    shoulder_in = shoulder_cm / CM_PER_INCH
    length_in = body_length_cm / CM_PER_INCH

    def nearest_index(measurement, reference_column):
        return min(
            range(len(TOP_SIZE_REFERENCE)),
            key=lambda index: abs(measurement - TOP_SIZE_REFERENCE[index][reference_column]),
        )

    # A garment's tag size can exceed the chest-only estimate for slim or
    # long-sleeved cuts, so use the largest size indicated by the measured fit
    # dimensions instead of letting one narrow chest measurement force size S.
    size_indices = [
        nearest_index(chest_in, 1),
        nearest_index(shoulder_in, 2),
        nearest_index(length_in, 3),
    ]
    sleeve_values = [value for value in (left_sleeve_cm, right_sleeve_cm) if value is not None]
    if sleeve_values:
        longest_sleeve_in = max(sleeve_values) / CM_PER_INCH
        if longest_sleeve_in >= 20.0:
            size_indices.append(nearest_index(longest_sleeve_in, 4))

    return labels[max(size_indices)]


def closest_pants_size(waist_width_cm, hip_width_cm):
    # Numeric pants sizes are waist-led; hip measurement is a fit check and
    # breaks close waist matches.
    measured_waist_in = waist_width_cm * 2.0 / CM_PER_INCH
    measured_hip_in = hip_width_cm * 2.0 / CM_PER_INCH
    best_size = None
    best_score = float("inf")

    for label, waist_in, hip_in in PANTS_SIZE_REFERENCE:
        score = 3.0 * (measured_waist_in - waist_in) ** 2 + (measured_hip_in - hip_in) ** 2
        if score < best_score:
            best_size, best_score = label, score
    return best_size


def closest_skirt_size(waist_width_cm, hip_width_cm):
    # Hip circumference is the main fit constraint for skirts; waist is the
    # secondary fit check. This reuses the project's existing bottom chart.
    measured_waist_in = waist_width_cm * 2.0 / CM_PER_INCH
    measured_hip_in = hip_width_cm * 2.0 / CM_PER_INCH
    best_size, best_score = None, float("inf")
    for label, waist_in, hip_in in PANTS_SIZE_REFERENCE:
        score = (measured_waist_in - waist_in) ** 2 + 3.0 * (measured_hip_in - hip_in) ** 2
        if score < best_score:
            best_size, best_score = label, score
    return best_size


def draw_size_label(image, size_text, label_prefix="Estimated India men's size"):
    text = f"{label_prefix}: {size_text}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.95
    thickness = 3
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    while text_w > image.shape[1] - 32 and scale > 0.5:
        scale -= 0.05
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = 16, text_h + 18
    cv2.rectangle(
        image,
        (max(0, x - 8), max(0, y - text_h - 8)),
        (min(image.shape[1] - 1, x + text_w + 8), min(image.shape[0] - 1, y + baseline + 6)),
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

    # Do not let a single horizontal slice reclassify a horizontally rotated
    # shirt as pants. Require both a pants-prompt mask and an upright silhouette
    # before using the leg-transition fallback.
    if active_mask is not None and best_bot_mask is not None:
        mask_u8 = (active_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            if h > w * 1.05:
                slice_y = int(y + h * 0.85)
                if slice_y < mask_u8.shape[0]:
                    row = mask_u8[slice_y, :]
                    diff = np.diff(row.astype(int))
                    transitions = np.count_nonzero(diff != 0)
                    if transitions >= 4:
                        garment_type = "pants"
                        label = "pants"

    return garment_type, active_mask, label


def classify_garment_with_variants(processor, state, image_shape):
    """Extend the original top/pants classifier with skirt and shorts prompts."""
    garment_type, garment_mask, detected_label = classify_garment(
        processor, state, image_shape
    )

    best_type, best_mask, best_label = garment_type, garment_mask, detected_label
    if garment_mask is not None:
        _, _, best_score = get_mask_and_box(
            processor, state, detected_label, image_shape
        )
    else:
        best_score = -1.0

    candidates = (
        ("skirt", ("skirt", "a skirt")),
        ("shorts", ("shorts", "short pants", "a pair of shorts")),
    )
    for candidate_type, prompts in candidates:
        candidate_mask = None
        candidate_score = -1.0
        for prompt in prompts:
            candidate_mask, _, candidate_score = get_mask_and_box(
                processor, state, prompt, image_shape
            )
            if candidate_mask is not None:
                break

        if candidate_mask is not None and candidate_score > best_score:
            best_type = candidate_type
            best_mask = candidate_mask
            best_label = candidate_type
            best_score = candidate_score

    return best_type, best_mask, best_label


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

                # Use a stable point along the shoulder slope rather than the
                # largest local contour angle, which often locks onto the neck
                # seam or a wrinkle on wide/uneven flat-lay shirts.
                target_x = neck_pt[0] + (pit_pt[0] - neck_pt[0]) * 0.80
                best_idx = int(np.argmin(np.abs(smooth_edge[:, 0] - target_x)))
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


def extract_top_landmarks_with_orientation(processor, state, pil_image, top_mask):
    """Try sideways orientations for ambiguous top masks, then map landmarks back."""
    ys, xs = np.where(top_mask)
    if len(xs) == 0:
        return None
    box_w = int(xs.max() - xs.min() + 1)
    box_h = int(ys.max() - ys.min() + 1)
    aspect = max(box_w, box_h) / max(1, min(box_w, box_h))

    # Rotation checks are useful when sleeves make a sideways top's bounding
    # box nearly square; skip clearly upright or wide silhouettes.
    if aspect > 1.25:
        return extract_top_landmarks(top_mask)

    def best_top_candidate(candidate_state, candidate_shape):
        best_mask, best_score = None, -1.0
        for prompt in ("shirt", "t-shirt"):
            candidate_mask, _, score = get_mask_and_box(
                processor, candidate_state, prompt, candidate_shape
            )
            if candidate_mask is not None and score > best_score:
                best_mask, best_score = candidate_mask, score
        return best_mask, best_score

    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                best_mask, best_score = best_top_candidate(
                    state, (pil_image.height, pil_image.width)
                )
                best_angle = 0
                for angle in (90, 270):
                    rotated_image = pil_image.rotate(angle, expand=True)
                    rotated_state = processor.set_image(rotated_image)
                    rotated_mask, rotated_score = best_top_candidate(
                        rotated_state, (rotated_image.height, rotated_image.width)
                    )
                    if rotated_mask is not None and rotated_score > best_score + 0.04:
                        best_mask, best_score, best_angle = (
                            rotated_mask, rotated_score, angle
                        )
        else:
            best_mask, best_score = best_top_candidate(
                state, (pil_image.height, pil_image.width)
            )
            best_angle = 0
            for angle in (90, 270):
                rotated_image = pil_image.rotate(angle, expand=True)
                rotated_state = processor.set_image(rotated_image)
                rotated_mask, rotated_score = best_top_candidate(
                    rotated_state, (rotated_image.height, rotated_image.width)
                )
                if rotated_mask is not None and rotated_score > best_score + 0.04:
                    best_mask, best_score, best_angle = (
                        rotated_mask, rotated_score, angle
                    )

    if best_mask is None:
        return extract_top_landmarks(top_mask)
    landmarks = extract_top_landmarks(best_mask)
    if landmarks is None or best_angle == 0:
        return landmarks

    original_width, original_height = pil_image.size

    def map_point(point):
        x, y = map(int, point)
        if best_angle == 90:  # PIL rotates counterclockwise.
            return original_width - 1 - y, x
        return y, original_height - 1 - x  # PIL 270 degrees is clockwise.

    return {name: map_point(point) for name, point in landmarks.items()}


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
        "waist_center": (
            int((waist_left[0] + waist_right[0]) / 2),
            int((waist_left[1] + waist_right[1]) / 2),
        ),
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


def extract_skirt_landmarks(skirt_mask):
    """Estimate skirt waistband, hip, and hem points from its segmented mask."""
    ys, xs = np.where(skirt_mask)
    if len(xs) == 0:
        return None

    top_y, bottom_y = int(ys.min()), int(ys.max())
    garment_height = max(1, bottom_y - top_y + 1)

    def horizontal_span(fraction):
        center_y = int(round(top_y + fraction * (garment_height - 1)))
        band_radius = max(2, int(round(garment_height * 0.015)))
        y0 = max(top_y, center_y - band_radius)
        y1 = min(bottom_y + 1, center_y + band_radius + 1)
        _, band_x = np.where(skirt_mask[y0:y1, :])
        if len(band_x) == 0:
            nearest_y = int(ys[np.argmin(np.abs(ys - center_y))])
            band_x = np.where(skirt_mask[nearest_y])[0]
            center_y = nearest_y
        if len(band_x) == 0:
            return None
        return (int(band_x.min()), center_y), (int(band_x.max()), center_y)

    waist = horizontal_span(0.05)
    hips = horizontal_span(0.40)
    if waist is None or hips is None:
        return None

    contours, _ = cv2.findContours(
        skirt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour_points = max(contours, key=cv2.contourArea).reshape(-1, 2)
    lower_points = contour_points[
        contour_points[:, 1] >= top_y + int(garment_height * 0.80)
    ]
    if len(lower_points) == 0:
        lower_points = contour_points[
            contour_points[:, 1] >= top_y + int(garment_height * 0.65)
        ]
    if len(lower_points) == 0:
        return None

    center_x = (int(xs.min()) + int(xs.max())) // 2
    left_lower = lower_points[lower_points[:, 0] <= center_x]
    right_lower = lower_points[lower_points[:, 0] >= center_x]
    if len(left_lower) == 0 or len(right_lower) == 0:
        return None

    hem_left = tuple(map(int, left_lower[np.argmin(left_lower[:, 0])]))
    hem_right = tuple(map(int, right_lower[np.argmax(right_lower[:, 0])]))
    top_center = ((waist[0][0] + waist[1][0]) // 2, waist[0][1])
    hem_center = (top_center[0], bottom_y)

    return {
        "waist_left": waist[0],
        "waist_right": waist[1],
        "hip_left": hips[0],
        "hip_right": hips[1],
        "hem_left": hem_left,
        "hem_right": hem_right,
        "top_center": top_center,
        "hem_center": hem_center,
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

    if not text:
        return

    mx = int((p1[0] + p2[0]) / 2) + offset[0]
    my = int((p1[1] + p2[1]) / 2) + offset[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Keep labels proportional to the source image. Small catalog images
    # otherwise get the same pixel-sized lettering as high-resolution photos.
    scale = float(np.clip(min(img.shape[:2]) / 1400.0 * 1.1, 0.45, 1.1))
    thickness = max(1, int(round(scale * 2.5)))
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    while text_w > img.shape[1] - 24 and scale > 0.55:
        scale -= 0.05
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

    x = max(8, min(mx, img.shape[1] - text_w - 8))
    y = max(text_h + 8, min(my, img.shape[0] - baseline - 8))
    background = img.copy()
    cv2.rectangle(
        background,
        (x - 5, y - text_h - 5),
        (min(img.shape[1] - 1, x + text_w + 5), min(img.shape[0] - 1, y + baseline + 5)),
        (15, 15, 15),
        -1,
    )
    cv2.addWeighted(background, 0.72, img, 0.28, 0, img)
    cv2.putText(img, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)


def draw_card_highlight(image, card_mask, card_rect):
    """Tint and outline the detected calibration card on the result image."""
    if card_mask is None or card_rect is None:
        return

    mask = card_mask.astype(bool)
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8), (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    overlay = image.copy()
    overlay[mask] = (0, 190, 255)  # Orange in OpenCV's BGR order.
    image[mask] = cv2.addWeighted(image[mask], 0.45, overlay[mask], 0.55, 0)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        cv2.drawContours(image, [max(contours, key=cv2.contourArea)], -1,
                         (0, 220, 255), 3, cv2.LINE_AA)
    box = cv2.boxPoints(card_rect).astype(np.int32)
    cv2.polylines(image, [box], True, (0, 255, 255), 2, cv2.LINE_AA)
    anchor_x = max(8, min(int(np.min(box[:, 0])), image.shape[1] - 8))
    anchor_y = max(30, min(int(np.min(box[:, 1])) - 8, image.shape[0] - 8))
    text = "Calibration card"
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = min(anchor_x, image.shape[1] - text_w - 12)
    y = max(text_h + 8, anchor_y)
    cv2.rectangle(image, (x - 5, y - text_h - 5),
                  (x + text_w + 5, y + baseline + 4), (15, 15, 15), -1)
    cv2.putText(image, text, (x, y), font, scale, (0, 255, 255),
                thickness, cv2.LINE_AA)


def measure_garment_auto(
    image_path: str,
    card_prompt: str = "credit card",
    output_dir: str = "./main_results"
):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target image not found: {image_path}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    pil_image = Image.open(image_path).convert("RGB")
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h, w, _ = cv_image.shape

    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = processor.set_image(pil_image)
                card_mask, card_rect, _ = get_mask_and_box(
                    processor, state, card_prompt, (h, w)
                )
                if card_mask is None and card_prompt != "card":
                    card_mask, card_rect, _ = get_mask_and_box(
                        processor, state, "card", (h, w)
                    )
                garment_type, cloth_mask, detected_label = classify_garment_with_variants(
                    processor, state, (h, w)
                )
        else:
            state = processor.set_image(pil_image)
            card_mask, card_rect, _ = get_mask_and_box(
                processor, state, card_prompt, (h, w)
            )
            if card_mask is None and card_prompt != "card":
                card_mask, card_rect, _ = get_mask_and_box(
                    processor, state, "card", (h, w)
                )
            garment_type, cloth_mask, detected_label = classify_garment_with_variants(
                processor, state, (h, w)
            )

    if cloth_mask is None:
        print("Error: Could not segment clothing item.")
        return

    ppm = None
    if card_mask is not None and card_rect is not None:
        (_, _), (dim_a, dim_b), _ = card_rect
        ppm = (
            max(dim_a, dim_b) / CARD_REAL_WIDTH_CM
            + min(dim_a, dim_b) / CARD_REAL_HEIGHT_CM
        ) / 2.0

    output_img = cv_image.copy()
    print(f"\n[Detected Category]: {garment_type.upper()} ({detected_label})")
    if ppm is None:
        print("[Calibration]: No card detected; reporting pixel distances only.")
    else:
        print(f"[Calibration PPM]:   {ppm:.2f} px/cm")

    if garment_type in {"pants", "shorts"}:
        lm = extract_pants_landmarks(cloth_mask)
        if lm is None:
            return

        waist = euclidean_dist_cm(lm["waist_left"], lm["waist_right"], ppm) if ppm else None
        hips = euclidean_dist_cm(lm["hip_left"], lm["hip_right"], ppm) if ppm else None
        front_rise = euclidean_dist_cm(lm["waist_center"], lm["crotch"], ppm) if ppm else None
        outseam = euclidean_dist_cm(lm["waist_left"], lm["left_cuff_out"], ppm) if ppm else None
        inseam = euclidean_dist_cm(lm["crotch"], lm["left_cuff_center"], ppm) if ppm else None
        cuff = euclidean_dist_cm(lm["left_cuff_out"], lm["left_cuff_in"], ppm) if ppm else None

        def display_value(point_a, point_b, calibrated_value):
            pixels = float(np.linalg.norm(np.asarray(point_a) - np.asarray(point_b)))
            return f"{calibrated_value:.1f} cm ({pixels:.0f} px)" if ppm else f"{pixels:.1f} px"

        print(f"\n--- {garment_type.title()} Measurements ---")
        if ppm:
            print(f"Waist Width:        {waist:.1f} cm (Half) | {waist * 2:.1f} cm (Full)")
            print(f"Hip Width:          {hips:.1f} cm")
            print(f"Front Rise:         {front_rise:.1f} cm")
            print(f"Outseam:            {outseam:.1f} cm")
            print(f"Inseam:             {inseam:.1f} cm")
            print(f"Cuff/Hem Width:     {cuff:.1f} cm")
            size_label = closest_pants_size(waist, hips)
            if garment_type == "pants":
                size_prefix = "Estimated India men's size"
                print(f"{size_prefix}: {size_label} (reference-chart match)")
            else:
                size_prefix = "Estimated bottom size (project reference chart)"
                print(f"{size_prefix}: {size_label} (waist/hip match)")
        else:
            print(f"Waist Width:        {display_value(lm['waist_left'], lm['waist_right'], 0)} (Half)")
            print(f"Hip Width:          {display_value(lm['hip_left'], lm['hip_right'], 0)}")
            print(f"Front Rise:         {display_value(lm['waist_center'], lm['crotch'], 0)}")
            print(f"Outseam:            {display_value(lm['waist_left'], lm['left_cuff_out'], 0)}")
            print(f"Inseam:             {display_value(lm['crotch'], lm['left_cuff_center'], 0)}")
            print(f"Cuff/Hem Width:     {display_value(lm['left_cuff_out'], lm['left_cuff_in'], 0)}")
            size_label = None
            size_prefix = None

        measurements = [
            ("Waist", "waist_left", "waist_right", waist, (0, 255, 255), (-60, -15)),
            ("Hips", "hip_left", "hip_right", hips, (255, 100, 255), (-50, -10)),
            ("Front rise", "waist_center", "crotch", front_rise, (0, 180, 255), (15, -5)),
            ("Outseam", "waist_left", "left_cuff_out", outseam, (0, 220, 0), (-150, 0)),
            ("Inseam", "crotch", "left_cuff_center", inseam, (0, 165, 255), (10, 0)),
            ("Cuff", "left_cuff_out", "left_cuff_in", cuff, (255, 50, 50), (-40, 25)),
        ]

    elif garment_type == "skirt":
        lm = extract_skirt_landmarks(cloth_mask)
        if lm is None:
            return
        waist = euclidean_dist_cm(lm["waist_left"], lm["waist_right"], ppm) if ppm else None
        hips = euclidean_dist_cm(lm["hip_left"], lm["hip_right"], ppm) if ppm else None
        hem = euclidean_dist_cm(lm["hem_left"], lm["hem_right"], ppm) if ppm else None
        length = euclidean_dist_cm(lm["top_center"], lm["hem_center"], ppm) if ppm else None

        def skirt_display(point_a, point_b, calibrated_value):
            pixels = float(np.linalg.norm(np.asarray(point_a) - np.asarray(point_b)))
            return f"{calibrated_value:.1f} cm ({pixels:.0f} px)" if ppm else f"{pixels:.1f} px"

        print("\n--- Skirt Measurements ---")
        if ppm:
            print(f"Waist Width (Half): {waist:.1f} cm")
            print(f"Hip Width:          {hips:.1f} cm")
            print(f"Hem Width:          {hem:.1f} cm")
            print(f"Skirt Length:       {length:.1f} cm")
            size_label = closest_skirt_size(waist, hips)
            size_prefix = "Estimated skirt size (project bottom chart)"
            print(f"{size_prefix}: {size_label} (waist/hip match)")
        else:
            print(f"Waist Width (Half): {skirt_display(lm['waist_left'], lm['waist_right'], 0)}")
            print(f"Hip Width:          {skirt_display(lm['hip_left'], lm['hip_right'], 0)}")
            print(f"Hem Width:          {skirt_display(lm['hem_left'], lm['hem_right'], 0)}")
            print(f"Skirt Length:       {skirt_display(lm['top_center'], lm['hem_center'], 0)}")
            size_label = None
            size_prefix = None

        measurements = [
            ("Waist", "waist_left", "waist_right", waist, (0, 255, 255), (0, -15)),
            ("Hips", "hip_left", "hip_right", hips, (255, 100, 255), (0, -10)),
            ("Hem", "hem_left", "hem_right", hem, (255, 50, 50), (0, 25)),
            ("Length", "top_center", "hem_center", length, (0, 220, 0), (10, 0)),
        ]

    else:
        lm = extract_top_landmarks_with_orientation(
            processor, state, pil_image, cloth_mask
        )
        if lm is None:
            return

        shoulders = euclidean_dist_cm(lm["left_shoulder"], lm["right_shoulder"], ppm) if ppm else None
        chest = euclidean_dist_cm(lm["left_armpit"], lm["right_armpit"], ppm) if ppm else None
        body_length = euclidean_dist_cm(lm["collar"], lm["hem"], ppm) if ppm else None
        left_sleeve = euclidean_dist_cm(lm["left_shoulder"], lm["left_cuff"], ppm) if ppm else None
        right_sleeve = euclidean_dist_cm(lm["right_shoulder"], lm["right_cuff"], ppm) if ppm else None

        def top_display(point_a, point_b, calibrated_value):
            pixels = float(np.linalg.norm(np.asarray(point_a) - np.asarray(point_b)))
            return f"{calibrated_value:.1f} cm ({pixels:.0f} px)" if ppm else f"{pixels:.1f} px"

        print("\n--- Top Measurements ---")
        if ppm:
            print(f"Shoulders:    {shoulders:.1f} cm")
            print(f"Chest Width:  {chest:.1f} cm")
            print(f"Body Length:  {body_length:.1f} cm")
            print(f"Left Sleeve:  {left_sleeve:.1f} cm")
            print(f"Right Sleeve: {right_sleeve:.1f} cm")
            size_label = closest_top_size(
                chest, shoulders, body_length, left_sleeve, right_sleeve
            )
            size_prefix = "Estimated India men's size"
            print(f"{size_prefix}: {size_label} (reference-chart match)")
        else:
            print(f"Shoulders:    {top_display(lm['left_shoulder'], lm['right_shoulder'], 0)}")
            print(f"Chest Width:  {top_display(lm['left_armpit'], lm['right_armpit'], 0)}")
            print(f"Body Length:  {top_display(lm['collar'], lm['hem'], 0)}")
            print(f"Left Sleeve:  {top_display(lm['left_shoulder'], lm['left_cuff'], 0)}")
            print(f"Right Sleeve: {top_display(lm['right_shoulder'], lm['right_cuff'], 0)}")
            size_label = None
            size_prefix = None

        measurements = [
            ("Shoulders", "left_shoulder", "right_shoulder", shoulders, (200, 50, 180), (-110, 28)),
            ("Chest", "left_armpit", "right_armpit", chest, (230, 160, 40), (-80, -12)),
            ("Body length", "collar", "hem", body_length, (0, 220, 0), (-190, 80)),
            ("Sleeve L", "left_shoulder", "left_cuff", left_sleeve, (255, 80, 80), (-80, -15)),
            ("Sleeve R", "right_shoulder", "right_cuff", right_sleeve, (255, 80, 80), (10, -15)),
        ]

    for label, point_a_key, point_b_key, calibrated_value, color, offset in measurements:
        point_a, point_b = lm[point_a_key], lm[point_b_key]
        pixel_value = float(np.linalg.norm(np.asarray(point_a) - np.asarray(point_b)))
        text = (
            f"{label}: {calibrated_value:.1f} cm ({pixel_value:.0f} px)"
            if ppm
            else f"{label}: {pixel_value:.1f} px"
        )
        draw_styled_measurement(output_img, point_a, point_b, text, color, offset=offset)

    draw_card_highlight(output_img, card_mask, card_rect)
    if size_label is not None:
        draw_size_label(output_img, size_label, size_prefix)
    out_file = os.path.join(output_dir, f"{base_name}_measurements.png")
    cv2.imwrite(out_file, output_img)
    print(f"\nResult saved to: {out_file}")


if __name__ == "__main__":
    measure_garment_auto("test_clothes/cloth11.jpeg")
