"""Utilities to map text-span detections to image-space bounding boxes.

The primary entry point is `map_text_spans_to_image_bboxes` which accepts
detections with character offsets (relative to a provided region text) and
per-word OCR boxes that include character offsets. It returns detections
enriched with a canonical `bbox` (x1,y1,x2,y2) and `page_number`.
"""
from typing import Any, Dict, List, Optional


def _union_boxes(boxes: List[List[int]]) -> Optional[List[int]]:
    if not boxes:
        return None
    xs1 = [int(b[0]) for b in boxes]
    ys1 = [int(b[1]) for b in boxes]
    xs2 = [int(b[2]) for b in boxes]
    ys2 = [int(b[3]) for b in boxes]
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end <= b_start or a_start >= b_end)


def map_text_spans_to_image_bboxes(
    detections: List[Dict[str, Any]],
    word_boxes: List[Dict[str, Any]],
    region_bbox: Optional[List[int]] = None,
    page_number: int = 1,
) -> List[Dict[str, Any]]:
    """Map detection char spans to a pixel bbox using OCR word boxes.

    Args:
        detections: list of dicts containing at least `start` and `end`
            (character offsets relative to the region text) and `text`.
        word_boxes: list of dicts with keys `text`, `bbox` and ideally
            `start_char`/`end_char` that map to the region text.
        region_bbox: fallback bbox if no precise word-level boxes match.
        page_number: page index for the returned detections.

    Returns:
        New list of detections where each detection has `bbox` set to
        [x1,y1,x2,y2] (integers) and `page_number` set.
    """
    out: List[Dict[str, Any]] = []

    # normalize word_boxes: ensure bbox lists and optional char spans
    normalized_words = []
    for w in word_boxes or []:
        try:
            bbox = list(map(int, w.get("bbox", [0, 0, 0, 0])))
        except Exception:
            bbox = [0, 0, 0, 0]
        start_c = w.get("start_char")
        end_c = w.get("end_char")
        normalized_words.append({"text": w.get("text", ""), "bbox": bbox, "start_char": start_c, "end_char": end_c})

    for d in detections or []:
        det = dict(d)
        # accept multiple possible field names for offsets
        start = det.get("start") if det.get("start") is not None else det.get("start_char")
        end = det.get("end") if det.get("end") is not None else det.get("end_char")
        try:
            start = int(start) if start is not None else None
            end = int(end) if end is not None else None
        except Exception:
            start = None
            end = None

        matched_boxes: List[List[int]] = []

        # 1) Prefer exact char-overlap mapping against word_boxes (deterministic)
        if start is not None and end is not None:
            for w in normalized_words:
                ws = w.get("start_char")
                we = w.get("end_char")
                if ws is None or we is None:
                    continue
                try:
                    if _overlap(start, end, int(ws), int(we)):
                        if w.get("bbox"):
                            matched_boxes.append(w.get("bbox"))
                except Exception:
                    continue

        # 2) If no char-span match, fall back to cautious substring token matching
        if not matched_boxes and normalized_words:
            det_text = (det.get("text") or "").strip().lower()
            if det_text:
                # attempt to find contiguous sequences of word tokens that best
                # match the detection text. This avoids accidental partial
                # matches spread across unrelated tokens.
                texts = [(w.get("text") or "").strip().lower() for w in normalized_words]
                n = len(texts)
                for i in range(n):
                    if not texts[i]:
                        continue
                    concat = texts[i]
                    boxes = [normalized_words[i].get("bbox")]
                    if concat == det_text or det_text in concat or concat in det_text:
                        matched_boxes.append(boxes[0]) if boxes[0] else None
                        break
                    for j in range(i + 1, n):
                        if not texts[j]:
                            continue
                        concat = concat + " " + texts[j]
                        boxes.append(normalized_words[j].get("bbox"))
                        if concat == det_text or det_text in concat or concat in det_text:
                            matched_boxes.extend([b for b in boxes if b])
                            break
                    if matched_boxes:
                        break

        # Aggregate matched boxes into union or fall back to region bbox
        bbox = _union_boxes(matched_boxes) if matched_boxes else (list(region_bbox) if region_bbox else None)
        if bbox:
            det["bbox"] = [int(x) for x in bbox]
        else:
            det.setdefault("bbox", region_bbox)
        det.setdefault("page_number", int(page_number or 1))
        out.append(det)

    return out
