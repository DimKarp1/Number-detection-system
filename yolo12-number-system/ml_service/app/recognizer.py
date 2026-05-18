from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


DETECT_MODEL_PATH = Path("models/number_detect_yolo12n.pt")
SEGMENT_MODEL_PATH = Path("models/number_segment_crop_yolo12.pt")
DIGIT_MODEL_PATH = Path("models/digit_detect_yolo12n.pt")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Device for YOLO inference.
# Values:
#   auto   - Ultralytics/PyTorch chooses automatically
#   cpu    - force CPU inference
#   cuda:0 - use first NVIDIA GPU, if CUDA is available
ML_DEVICE = os.getenv("ML_DEVICE", "auto").strip().lower()


def predict_device_kwargs() -> dict[str, str]:
    if ML_DEVICE in {"", "auto"}:
        return {}

    return {"device": ML_DEVICE}

DEFAULT_DETECT_CONF = 0.12
DEFAULT_SEGMENT_CONF = 0.17
DEFAULT_DIGIT_CONF = 0.3

DETECT_IMGSZ = 960
SEGMENT_IMGSZ = 416
DIGIT_IMGSZ = 416
#512

CROP_MARGIN_RATIO = 0.40
QUAD_EXPAND_SCALE = 1.15

FALSE_ONE_MIN_CONF = 0.75

MIN_RECOGNIZED_AVG_DIGIT_CONF = 0.70
MIN_RECOGNIZED_DIGITS = 1
MAX_RECOGNIZED_DIGITS = 3

MIN_PLATE_ASPECT = 1.10
MIN_DIGIT_SPAN_RATIO = 0.22

DUPLICATE_IOU_THRESHOLD = 0.35
DUPLICATE_CONTAINMENT_THRESHOLD = 0.70
SAME_NUMBER_IOU_THRESHOLD = 0.15


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images(source: Path, limit: int | None = None) -> list[Path]:
    if source.is_file():
        return [source]

    images = sorted(
        p for p in source.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
    )

    if limit is not None:
        images = images[:limit]

    return images


def expand_bbox(
    xyxy: np.ndarray,
    img_w: int,
    img_h: int,
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy.astype(float)

    bw = x2 - x1
    bh = y2 - y1

    mx = bw * margin_ratio
    my = bh * margin_ratio

    nx1 = int(max(0, x1 - mx))
    ny1 = int(max(0, y1 - my))
    nx2 = int(min(img_w, x2 + mx))
    ny2 = int(min(img_h, y2 + my))

    if nx2 <= nx1:
        nx2 = min(img_w, nx1 + 1)

    if ny2 <= ny1:
        ny2 = min(img_h, ny1 + 1)

    return nx1, ny1, nx2, ny2


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0

    return abs(cv2.contourArea(poly.astype(np.float32)))


def get_largest_mask_polygon(seg_result: Any) -> tuple[np.ndarray | None, float | None]:
    if seg_result.masks is None or seg_result.boxes is None:
        return None, None

    polygons = seg_result.masks.xy
    if not polygons:
        return None, None

    confs = (
        seg_result.boxes.conf.cpu().numpy()
        if seg_result.boxes.conf is not None
        else None
    )

    best_idx = -1
    best_area = -1.0

    for i, poly in enumerate(polygons):
        poly_np = np.asarray(poly, dtype=np.float32)
        area = polygon_area(poly_np)

        if area > best_area:
            best_area = area
            best_idx = i

    if best_idx < 0:
        return None, None

    best_poly = np.asarray(polygons[best_idx], dtype=np.float32)
    best_conf = (
        float(confs[best_idx])
        if confs is not None and len(confs) > best_idx
        else None
    )

    return best_poly, best_conf


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Возвращает 4 точки в порядке:
    top-left, top-right, bottom-right, bottom-left.
    """
    pts = pts.astype(np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def polygon_to_quad(poly: np.ndarray) -> np.ndarray:
    """
    Превращает произвольный полигон маски в 4-точечный повернутый прямоугольник.
    """
    contour = poly.astype(np.float32)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)

    return order_points(box)


def expand_quad_from_center(quad: np.ndarray, scale: float) -> np.ndarray:
    """
    Немного расширяет четырехугольник от центра.
    Это помогает не срезать края цифр при выравнивании.
    """
    center = quad.mean(axis=0)
    expanded = (quad - center) * scale + center

    return expanded.astype(np.float32)


def warp_quad(
    image: np.ndarray,
    quad: np.ndarray,
    force_horizontal: bool = True,
) -> tuple[np.ndarray, bool]:
    """
    Делает perspective warp по quad.
    """
    ordered = order_points(quad)

    tl, tr, br, bl = ordered

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_w = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_h = int(max(height_a, height_b))

    max_w = max(max_w, 8)
    max_h = max(max_h, 8)

    dst = np.array(
        [
            [0, 0],
            [max_w - 1, 0],
            [max_w - 1, max_h - 1],
            [0, max_h - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, matrix, (max_w, max_h))

    rotated_to_horizontal = False

    if force_horizontal:
        h, w = warped.shape[:2]

        if h > w:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            rotated_to_horizontal = True

    return warped, rotated_to_horizontal


def rotate_180(image: np.ndarray) -> np.ndarray:
    return cv2.rotate(image, cv2.ROTATE_180)


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def bbox_intersection_stats(
    box_a: list[float] | np.ndarray,
    box_b: list[float] | np.ndarray,
) -> tuple[float, float]:
    """
    Возвращает:
    - IoU;
    - долю пересечения относительно меньшего bbox.

    Второй показатель полезен, когда один bbox почти вложен в другой.
    """
    a = np.array(box_a, dtype=np.float32)
    b = np.array(box_b, dtype=np.float32)

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    iou = inter_area / union if union > 0 else 0.0

    min_area = min(area_a, area_b)
    containment = inter_area / min_area if min_area > 0 else 0.0

    return float(iou), float(containment)


def get_digits_span_ratio(candidate: dict[str, Any]) -> float:
    """
    Оценивает, какую часть plate по ширине занимает группа найденных цифр.
    Если цифры занимают очень маленький участок, это часто мусор.
    """
    digits = candidate.get("digits", [])
    plate_size = candidate.get("plate_size", None)

    if not digits or not plate_size:
        return 0.0

    plate_w = float(plate_size[0])

    if plate_w <= 0:
        return 0.0

    xs1 = []
    xs2 = []

    for digit in digits:
        box = digit.get("bbox_xyxy", [])
        if len(box) != 4:
            continue

        xs1.append(float(box[0]))
        xs2.append(float(box[2]))

    if not xs1 or not xs2:
        return 0.0

    span = max(xs2) - min(xs1)

    return float(span / plate_w)


def is_candidate_recognized(candidate: dict[str, Any]) -> bool:
    """
    Фильтр кандидатов перед попаданием в recognized_numbers.
    """
    number = candidate.get("number", "")
    digits = candidate.get("digits", [])
    avg_conf = float(candidate.get("avg_digit_confidence", 0.0))
    plate_aspect = float(candidate.get("plate_aspect", 0.0))
    skip_reason = candidate.get("skip_reason")

    if skip_reason:
        return False

    if not number:
        return False

    if len(number) < MIN_RECOGNIZED_DIGITS:
        return False

    if len(number) > MAX_RECOGNIZED_DIGITS:
        return False

    if len(digits) != len(number):
        return False

    if avg_conf < MIN_RECOGNIZED_AVG_DIGIT_CONF:
        return False

    if plate_aspect < MIN_PLATE_ASPECT:
        return False

    digit_span_ratio = get_digits_span_ratio(candidate)
    if digit_span_ratio < MIN_DIGIT_SPAN_RATIO:
        return False

    return True


def deduplicate_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Удаляет повторные распознавания одного и того же номера.

    Логика:
    - кандидаты уже должны быть отсортированы по candidate_score убыванию;
    - если новый bbox сильно пересекается с уже оставленным, считаем дублем;
    - если номер одинаковый и bbox хотя бы немного пересекается, тоже считаем дублем.
    """
    kept: list[dict[str, Any]] = []

    for candidate in candidates:
        if not is_candidate_recognized(candidate):
            continue

        candidate_box = candidate.get("detect_bbox_xyxy")
        candidate_number = candidate.get("number", "")

        if not candidate_box:
            continue

        is_duplicate = False

        for kept_candidate in kept:
            kept_box = kept_candidate.get("detect_bbox_xyxy")
            kept_number = kept_candidate.get("number", "")

            if not kept_box:
                continue

            iou, containment = bbox_intersection_stats(candidate_box, kept_box)

            # Один и тот же физический объект с разными границами.
            if iou >= DUPLICATE_IOU_THRESHOLD:
                is_duplicate = True
                break

            # Один bbox почти вложен в другой.
            if containment >= DUPLICATE_CONTAINMENT_THRESHOLD:
                is_duplicate = True
                break

            # Тот же номер рядом/почти там же.
            if candidate_number == kept_number and iou >= SAME_NUMBER_IOU_THRESHOLD:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(candidate)

    return kept


def deduplicate_recognized_numbers_by_value(
    recognized_numbers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Если один и тот же номер найден несколько раз на одном фото,
    оставляем вариант с максимальным candidate_score.
    """
    best_by_number: dict[str, dict[str, Any]] = {}

    for item in recognized_numbers:
        number = item.get("number", "")
        score = float(item.get("candidate_score", -9999.0))

        if not number:
            continue

        current = best_by_number.get(number)

        if current is None:
            best_by_number[number] = item
            continue

        current_score = float(current.get("candidate_score", -9999.0))

        if score > current_score:
            best_by_number[number] = item

    result = list(best_by_number.values())

    result = sorted(
        result,
        key=lambda item: item["bbox_xyxy"][0],
    )

    return result


def suppress_duplicate_digits(
    boxes_xyxy: np.ndarray,
    classes: np.ndarray,
    confs: np.ndarray,
    iou_threshold: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(boxes_xyxy) <= 1:
        return boxes_xyxy, classes, confs

    order = np.argsort(-confs)
    keep_indices: list[int] = []

    for idx in order:
        box = boxes_xyxy[idx]

        duplicate = False

        for kept_idx in keep_indices:
            kept_box = boxes_xyxy[kept_idx]

            if bbox_iou(box, kept_box) >= iou_threshold:
                duplicate = True
                break

        if not duplicate:
            keep_indices.append(int(idx))

    keep_indices_np = np.array(keep_indices, dtype=int)

    return (
        boxes_xyxy[keep_indices_np],
        classes[keep_indices_np],
        confs[keep_indices_np],
    )


def filter_digit_boxes(
    boxes_xyxy: np.ndarray,
    classes: np.ndarray,
    confs: np.ndarray,
    image_shape: tuple[int, int, int],
    min_conf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Фильтрация лишних цифр после YOLO:
    - по confidence;
    - по размеру;
    - по дублям;
    - по положению на одной строке;
    - по похожести высоты;
    - отдельное ужесточение для ложной "1".
    """
    if len(boxes_xyxy) == 0:
        return boxes_xyxy, classes, confs

    img_h, img_w = image_shape[:2]

    widths = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

    keep = confs >= min_conf

    keep &= heights >= max(6, img_h * 0.18)
    keep &= heights <= img_h * 0.95
    keep &= widths >= max(3, img_w * 0.015)
    keep &= widths <= img_w * 0.65

    # Частая проблема: ложная "1" на вертикальных линиях/краях.
    # Поэтому для класса 1 требуем чуть более высокую уверенность.
    one_min_conf = max(min_conf, FALSE_ONE_MIN_CONF)
    for i in range(len(classes)):
        if classes[i] == 1 and confs[i] < one_min_conf:
            keep[i] = False

    if keep.sum() == 0:
        return boxes_xyxy[:0], classes[:0], confs[:0]

    boxes_xyxy = boxes_xyxy[keep]
    classes = classes[keep]
    confs = confs[keep]

    boxes_xyxy, classes, confs = suppress_duplicate_digits(
        boxes_xyxy=boxes_xyxy,
        classes=classes,
        confs=confs,
        iou_threshold=0.45,
    )

    if len(boxes_xyxy) == 0:
        return boxes_xyxy, classes, confs

    heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    y_centers = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0

    median_h = float(np.median(heights))
    median_y = float(np.median(y_centers))

    keep = np.abs(y_centers - median_y) <= median_h * 0.75
    keep &= heights >= median_h * 0.45
    keep &= heights <= median_h * 1.75

    boxes_xyxy = boxes_xyxy[keep]
    classes = classes[keep]
    confs = confs[keep]

    if len(boxes_xyxy) == 0:
        return boxes_xyxy, classes, confs

    x_centers = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
    order = np.argsort(x_centers)

    return boxes_xyxy[order], classes[order], confs[order]


def draw_digit_predictions(
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    out = image.copy()

    for box, cls_id in zip(boxes_xyxy, classes):
        x1, y1, x2, y2 = box.astype(int)
        label = str(int(cls_id))

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 1)

        cv2.putText(
            out,
            label,
            (x1, max(12, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return out


def score_digit_sequence(number: str, digits: list[dict[str, Any]]) -> float:
    if not digits:
        return -9999.0

    avg_conf = float(np.mean([d["confidence"] for d in digits]))

    score = 0.0

    # Не слишком агрессивно поощряем длину, чтобы лишние цифры не выигрывали только за счёт количества.
    score += len(number) * 6.0
    score += avg_conf * 8.0

    if len(number) < 1:
        score -= 100.0

    # Для текущего мотогоночного набора слишком длинные строки часто означают, что digit detector поймал мусор.
    if len(number) > 4:
        score -= (len(number) - 4) * 8.0

    if avg_conf < 0.45:
        score -= 20.0

    weak_digits = [d for d in digits if d["confidence"] < 0.45]
    score -= len(weak_digits) * 5.0

    return score


def recognize_digits_single(
    digit_model: YOLO,
    plate_img: np.ndarray,
    conf: float,
) -> tuple[str, list[dict[str, Any]], np.ndarray, float]:
    result = digit_model.predict(
        source=plate_img,
        imgsz=DIGIT_IMGSZ,
        conf=conf,
        iou=0.45,
        verbose=False,
        **predict_device_kwargs(),
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        return "", [], plate_img.copy(), -9999.0

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    boxes_xyxy, classes, confs = filter_digit_boxes(
        boxes_xyxy=boxes_xyxy,
        classes=classes,
        confs=confs,
        image_shape=plate_img.shape,
        min_conf=conf,
    )

    if len(boxes_xyxy) == 0:
        return "", [], plate_img.copy(), -9999.0

    number = "".join(str(int(c)) for c in classes)

    digits: list[dict[str, Any]] = []

    for box, cls_id, digit_conf in zip(boxes_xyxy, classes, confs):
        digits.append(
            {
                "digit": str(int(cls_id)),
                "confidence": float(digit_conf),
                "bbox_xyxy": [float(v) for v in box],
            }
        )

    debug_img = draw_digit_predictions(plate_img, boxes_xyxy, classes)
    digit_score = score_digit_sequence(number, digits)

    return number, digits, debug_img, digit_score


def recognize_digits_with_optional_180(
    digit_model: YOLO,
    plate_img: np.ndarray,
    conf: float,
    allow_180: bool,
) -> tuple[str, list[dict[str, Any]], np.ndarray, float, str, list[dict[str, Any]], dict[str, np.ndarray]]:
    """
    Обычный plate проверяется только в ориентации 0°.

    Если allow_180=True, дополнительно проверяется вариант 180°.
    Возвращает лучший результат и debug-картинки всех проверенных ориентаций.
    """
    attempts: list[dict[str, Any]] = []
    debug_images: dict[str, np.ndarray] = {}

    number_0, digits_0, debug_0, score_0 = recognize_digits_single(
        digit_model=digit_model,
        plate_img=plate_img,
        conf=conf,
    )

    debug_images["0"] = debug_0

    attempts.append(
        {
            "orientation": "0",
            "number": number_0,
            "score": float(score_0),
            "digits_count": len(digits_0),
            "avg_digit_confidence": float(np.mean([d["confidence"] for d in digits_0])) if digits_0 else 0.0,
        }
    )

    best_number = number_0
    best_digits = digits_0
    best_debug = debug_0
    best_score = score_0
    best_orientation = "0"

    if allow_180:
        plate_180 = rotate_180(plate_img)

        number_180, digits_180, debug_180, score_180 = recognize_digits_single(
            digit_model=digit_model,
            plate_img=plate_180,
            conf=conf,
        )

        debug_images["180"] = debug_180

        attempts.append(
            {
                "orientation": "180",
                "number": number_180,
                "score": float(score_180),
                "digits_count": len(digits_180),
                "avg_digit_confidence": float(np.mean([d["confidence"] for d in digits_180])) if digits_180 else 0.0,
            }
        )

        if score_180 > best_score:
            best_number = number_180
            best_digits = digits_180
            best_debug = debug_180
            best_score = score_180
            best_orientation = "180"

    return best_number, best_digits, best_debug, best_score, best_orientation, attempts, debug_images


def compute_candidate_score(
    number: str,
    digits: list[dict[str, Any]],
    det_conf: float,
    seg_conf: float | None,
    plate_aspect: float,
) -> tuple[float, float]:
    avg_digit_conf = (
        float(np.mean([d["confidence"] for d in digits]))
        if digits
        else 0.0
    )

    score = (
        len(number) * 6.0
        + avg_digit_conf * 3.0
        + det_conf
        + (seg_conf or 0.0)
    )

    if len(number) < 2:
        score -= 100.0

    if len(number) > 4:
        score -= (len(number) - 4) * 8.0

    if avg_digit_conf < 0.45:
        score -= 15.0

    if plate_aspect < 1.15:
        score -= 8.0

    return score, avg_digit_conf


def draw_original_debug(
    image: np.ndarray,
    candidates: list[dict[str, Any]],
) -> np.ndarray:
    out = image.copy()

    for candidate in candidates:
        box = np.array(candidate["detect_bbox_xyxy"], dtype=np.float32).astype(int)
        x1, y1, x2, y2 = box

        number = candidate.get("number", "")
        score = candidate.get("candidate_score", 0.0)
        skipped = candidate.get("skip_reason")

        if skipped:
            color = (128, 128, 128)
            label = f"skip:{skipped}"
        elif len(number) >= 2:
            color = (0, 255, 0)
            label = f"{number} s={score:.1f}"
        else:
            color = (0, 0, 255)
            label = f"no-number s={score:.1f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            out,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return out


def process_image(
    image_path: Path,
    out_dir: Path,
    detect_model: YOLO,
    segment_model: YOLO,
    digit_model: YOLO,
    detect_conf: float,
    segment_conf: float,
    digit_conf: float,
    max_candidates: int,
) -> dict[str, Any]:
    image = cv2.imread(str(image_path))

    if image is None:
        return {
            "image": str(image_path),
            "error": "failed_to_read_image",
            "recognized_numbers": [],
            "candidates": [],
        }

    img_h, img_w = image.shape[:2]

    stem = image_path.stem
    image_out_dir = out_dir / stem
    ensure_dir(image_out_dir)

    detect_result = detect_model.predict(
        source=image,
        imgsz=DETECT_IMGSZ,
        conf=detect_conf,
        iou=0.7,
        verbose=False,
        **predict_device_kwargs(),
    )[0]

    candidates: list[dict[str, Any]] = []

    if detect_result.boxes is None or len(detect_result.boxes) == 0:
        cv2.imwrite(str(image_out_dir / "original_no_detections.jpg"), image)

        return {
            "image": str(image_path),
            "recognized_numbers": [],
            "candidates": [],
        }

    det_boxes = detect_result.boxes.xyxy.cpu().numpy()
    det_confs = detect_result.boxes.conf.cpu().numpy()

    det_order = np.argsort(-det_confs)
    det_order = det_order[:max_candidates]

    for rank, idx in enumerate(det_order):
        det_box = det_boxes[idx]
        det_conf_value = float(det_confs[idx])

        x1, y1, x2, y2 = expand_bbox(
            xyxy=det_box,
            img_w=img_w,
            img_h=img_h,
            margin_ratio=CROP_MARGIN_RATIO,
        )

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        candidate_prefix = f"candidate_{rank:02d}"

        cv2.imwrite(
            str(image_out_dir / f"{candidate_prefix}_detect_crop.jpg"),
            crop,
        )

        seg_result = segment_model.predict(
            source=crop,
            imgsz=SEGMENT_IMGSZ,
            conf=segment_conf,
            iou=0.7,
            verbose=False,
            **predict_device_kwargs(),
        )[0]

        poly, seg_conf_value = get_largest_mask_polygon(seg_result)

        if poly is None:
            candidates.append(
                {
                    "rank": rank,
                    "detect_confidence": det_conf_value,
                    "detect_bbox_xyxy": [float(v) for v in det_box],
                    "expanded_crop_xyxy": [x1, y1, x2, y2],
                    "segment_found": False,
                    "segment_confidence": None,
                    "number": "",
                    "digits": [],
                    "orientation": "none",
                    "orientation_attempts": [],
                    "plate_rotated_to_horizontal": False,
                    "plate_aspect": 0.0,
                    "plate_size": [0, 0],
                    "avg_digit_confidence": 0.0,
                    "candidate_score": -9999.0,
                    "skip_reason": "no_segment",
                }
            )
            continue

        crop_seg_debug = crop.copy()
        poly_int = poly.astype(np.int32).reshape((-1, 1, 2))

        cv2.polylines(
            crop_seg_debug,
            [poly_int],
            isClosed=True,
            color=(0, 0, 255),
            thickness=2,
        )

        cv2.imwrite(
            str(image_out_dir / f"{candidate_prefix}_segment_crop.jpg"),
            crop_seg_debug,
        )

        # Сегментация нужна только для получения quad.
        quad = polygon_to_quad(poly)
        quad = expand_quad_from_center(quad, scale=QUAD_EXPAND_SCALE)

        plate, rotated_to_horizontal = warp_quad(
            image=crop,
            quad=quad,
            force_horizontal=True,
        )

        plate_h, plate_w = plate.shape[:2]
        plate_aspect = plate_w / max(1, plate_h)

        cv2.imwrite(
            str(image_out_dir / f"{candidate_prefix}_plate_warped.jpg"),
            plate,
        )

        
        number, digits, digit_debug, digit_sequence_score, digit_orientation, orientation_attempts, orientation_debug_images = recognize_digits_with_optional_180(
            digit_model=digit_model,
            plate_img=plate,
            conf=digit_conf,
            allow_180=rotated_to_horizontal,
        )


        for orientation_name, debug_image in orientation_debug_images.items():
            cv2.imwrite(
                str(image_out_dir / f"{candidate_prefix}_digits_{orientation_name}.jpg"),
                debug_image,
            )

        cv2.imwrite(
            str(image_out_dir / f"{candidate_prefix}_digits_SELECTED_{digit_orientation}.jpg"),
            digit_debug,
        )

        # Для отладки сохраняем альтернативную 180-версию изображения, если такая проверка была разрешена.
        if rotated_to_horizontal:
            plate_180 = rotate_180(plate)
            cv2.imwrite(
                str(image_out_dir / f"{candidate_prefix}_plate_warped_180_debug.jpg"),
                plate_180,
            )

        candidate_score, avg_digit_conf = compute_candidate_score(
            number=number,
            digits=digits,
            det_conf=det_conf_value,
            seg_conf=seg_conf_value,
            plate_aspect=plate_aspect,
        )

        candidates.append(
            {
                "rank": rank,
                "detect_confidence": det_conf_value,
                "detect_bbox_xyxy": [float(v) for v in det_box],
                "expanded_crop_xyxy": [x1, y1, x2, y2],
                "segment_found": True,
                "segment_confidence": seg_conf_value,
                "number": number,
                "digits": digits,
                "orientation": digit_orientation,
                "orientation_attempts": orientation_attempts,
                "plate_rotated_to_horizontal": rotated_to_horizontal,
                "plate_aspect": plate_aspect,
                "plate_size": [int(plate_w), int(plate_h)],
                "digit_sequence_score": digit_sequence_score,
                "avg_digit_confidence": avg_digit_conf,
                "candidate_score": candidate_score,
                "skip_reason": None,
            }
        )

    candidates = sorted(
        candidates,
        key=lambda c: c.get("candidate_score", -9999.0),
        reverse=True,
    )

    deduplicated_candidates = deduplicate_candidates(candidates)

    recognized_numbers = []

    for candidate in deduplicated_candidates:
        recognized_numbers.append(
            {
                "number": candidate.get("number", ""),
                "avg_digit_confidence": candidate.get("avg_digit_confidence", 0.0),
                "detect_confidence": candidate.get("detect_confidence"),
                "segment_confidence": candidate.get("segment_confidence"),
                "orientation": candidate.get("orientation"),
                "orientation_attempts": candidate.get("orientation_attempts"),
                "plate_rotated_to_horizontal": candidate.get("plate_rotated_to_horizontal"),
                "plate_aspect": candidate.get("plate_aspect"),
                "plate_size": candidate.get("plate_size"),
                "bbox_xyxy": candidate.get("detect_bbox_xyxy"),
                "candidate_score": candidate.get("candidate_score"),
                "digit_span_ratio": get_digits_span_ratio(candidate),
            }
        )

    recognized_numbers = sorted(
        recognized_numbers,
        key=lambda item: item["bbox_xyxy"][0],
    )

    recognized_numbers = deduplicate_recognized_numbers_by_value(recognized_numbers)


    original_debug = draw_original_debug(image, candidates)
    cv2.imwrite(str(image_out_dir / "original_debug.jpg"), original_debug)

    return {
        "image": str(image_path),
        "recognized_numbers": recognized_numbers,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        help="Путь к изображению или папке с изображениями",
    )
    parser.add_argument(
        "--out",
        default="runs/e2e/pipeline_debug",
        help="Папка для сохранения результата",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничение числа изображений",
    )
    parser.add_argument(
        "--detect-conf",
        type=float,
        default=DEFAULT_DETECT_CONF,
    )
    parser.add_argument(
        "--segment-conf",
        type=float,
        default=DEFAULT_SEGMENT_CONF,
    )
    parser.add_argument(
        "--digit-conf",
        type=float,
        default=DEFAULT_DIGIT_CONF,
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    source = Path(args.source)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    if not DETECT_MODEL_PATH.exists():
        raise FileNotFoundError(DETECT_MODEL_PATH)

    if not SEGMENT_MODEL_PATH.exists():
        raise FileNotFoundError(SEGMENT_MODEL_PATH)

    if not DIGIT_MODEL_PATH.exists():
        raise FileNotFoundError(DIGIT_MODEL_PATH)

    print("Загружаю модели...")
    detect_model = YOLO(str(DETECT_MODEL_PATH))
    segment_model = YOLO(str(SEGMENT_MODEL_PATH))
    digit_model = YOLO(str(DIGIT_MODEL_PATH))

    images = list_images(source, limit=args.limit)

    print(f"Найдено изображений: {len(images)}")

    all_results = []

    for i, image_path in enumerate(images, start=1):
        print(f"[{i}/{len(images)}] {image_path}")

        result = process_image(
            image_path=image_path,
            out_dir=out_dir,
            detect_model=detect_model,
            segment_model=segment_model,
            digit_model=digit_model,
            detect_conf=args.detect_conf,
            segment_conf=args.segment_conf,
            digit_conf=args.digit_conf,
            max_candidates=args.max_candidates,
        )

        recognized = result.get("recognized_numbers", [])
        numbers = [item["number"] for item in recognized]

        print(f"  recognized_numbers: {numbers}")

        all_results.append(result)

    json_path = out_dir / "results.json"
    json_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nГотово.")
    print(f"Результаты: {out_dir}")
    print(f"JSON:       {json_path}")


if __name__ == "__main__":
    main()
