"""Utilities to map text-span detections to image-space bounding boxes.

The primary entry point is `map_text_spans_to_image_bboxes` which accepts
detections with character offsets (relative to a provided region text) and
per-word OCR boxes that include character offsets. It returns detections
enriched with a canonical `bbox` (x1,y1,x2,y2) and `page_number`.
"""
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TokenAlignment:
    orig: str
    new: str

    def __post_init__(self):
        # Build index maps using a conservative tokenizer (whitespace + punctuation)
        self.orig_tokens = self._tokenize_with_spans(self.orig)
        self.new_tokens = self._tokenize_with_spans(self.new)
        # Compute mapping from orig char index -> new char index using token alignment
        self.orig_to_new = self._build_char_map()

    def _tokenize_with_spans(self, s: str) -> List[Dict[str, Any]]:
        import re

        tokens: List[Dict[str, Any]] = []
        for m in re.finditer(r"\S+", s or ""):
            tokens.append({"text": m.group(0), "start": m.start(), "end": m.end()})
        return tokens

    def _build_char_map(self) -> List[int]:
        # Initialize with conservative mapping: map each orig char to the same index if possible
        orig_len = len(self.orig or "")
        new_len = len(self.new or "")
        orig_to_new = [0] * (orig_len + 1)

        # Greedy token alignment by normalized token text
        i_new = 0
        for t_orig in self.orig_tokens:
            found = False
            for j in range(i_new, len(self.new_tokens)):
                if self._norm(t_orig["text"]) == self._norm(self.new_tokens[j]["text"]):
                    # align token-level: map each char in orig token range to corresponding new token start offset
                    new_token_start = self.new_tokens[j]["start"]
                    for idx in range(t_orig["start"], t_orig["end"] + 1):
                        # clamp within new length
                        orig_to_new[idx] = min(new_len, new_token_start + (idx - t_orig["start"]))
                    i_new = j + 1
                    found = True
                    break
            if not found:
                # token not found; conservatively map to current new index
                for idx in range(t_orig["start"], t_orig["end"] + 1):
                    orig_to_new[idx] = min(new_len, i_new and self.new_tokens[i_new - 1]["end"] or 0)

        # final sentinel mapping
        orig_to_new[len(self.orig or "")] = len(self.new or "")
        # fill gaps deterministically
        last = 0
        for i in range(len(orig_to_new)):
            if orig_to_new[i] == 0 and i != 0:
                orig_to_new[i] = last
            else:
                last = orig_to_new[i]

        return orig_to_new

    def _norm(self, s: str) -> str:
        try:
            return s.strip().casefold()
        except Exception:
            return (s or "").strip().lower()

    def map_span(self, s: int, e: int) -> Tuple[int, int]:
        s = max(0, min(len(self.orig or ""), int(s or 0)))
        e = max(0, min(len(self.orig or ""), int(e or 0)))
        new_s = int(self.orig_to_new[s])
        if e > 0:
            new_e = int(self.orig_to_new[max(0, e - 1)]) + 1
        else:
            new_e = int(self.orig_to_new[0])
        return new_s, new_e



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


class MappingError(Exception):
    """Raised when a detection cannot be mapped to an image bbox."""
    pass


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

    # helper: case-fold normalization for robust comparisons
    def _norm(s: str) -> str:
        try:
            return (s or "").casefold().strip()
        except Exception:
            return (s or "").lower().strip()

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

        # 1) Fast path: Prefer exact char-overlap mapping against word_boxes (deterministic)
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
            det_text = _norm(det.get("text") or "")
            if det_text:
                # attempt to find contiguous sequences of word tokens that best
                # match the detection text; use casefolded comparison to handle
                # case and unicode normalization differences.
                texts = [_norm(w.get("text") or "") for w in normalized_words]
                n = len(texts)
                for i in range(n):
                    if not texts[i]:
                        continue
                    concat = texts[i]
                    boxes = [normalized_words[i].get("bbox")]
                    if concat == det_text or det_text in concat or concat in det_text:
                        if boxes[0]:
                            matched_boxes.append(boxes[0])
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

    # Aggregate matched boxes into union.
    bbox = _union_boxes(matched_boxes) if matched_boxes else None
    if bbox:
        det["bbox"] = [int(x) for x in bbox]
    else:
        # no fine-grained mapping found; consult SETTINGS.mapping.raise_on_fail
        try:
            from src.core.config import SETTINGS
        except Exception:
            SETTINGS = None

        raise_on_fail = False
        try:
            if SETTINGS is not None and getattr(SETTINGS, "mapping", None) is not None:
                raise_on_fail = bool(getattr(SETTINGS.mapping, "raise_on_fail", False))
        except Exception:
            raise_on_fail = False

        if raise_on_fail:
            raise MappingError(f"Failed to map detection '{det.get('text')}' start={start} end={end} to word boxes; region_bbox={region_bbox}")
        det.setdefault("bbox", region_bbox)
    det.setdefault("page_number", int(page_number or 1))
    out.append(det)

    return out
