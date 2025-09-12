from src.ocr.mapping import map_text_spans_to_image_bboxes


def test_map_multiword_span_union():
    # sample text and word boxes
    # "Patient Name: Emily R. Dawson"
    word_boxes = [
        {"text": "Patient", "bbox": [0, 0, 50, 10], "start_char": 0, "end_char": 7},
        {"text": "Name:", "bbox": [51, 0, 100, 10], "start_char": 8, "end_char": 13},
        {"text": "Emily", "bbox": [101, 0, 150, 10], "start_char": 14, "end_char": 19},
        {"text": "R.", "bbox": [151, 0, 165, 10], "start_char": 20, "end_char": 22},
        {"text": "Dawson", "bbox": [166, 0, 220, 10], "start_char": 23, "end_char": 29},
    ]

    detections = [{"text": "Emily R. Dawson", "start": 14, "end": 29}]
    mapped = map_text_spans_to_image_bboxes(detections, word_boxes, region_bbox=[0,0,220,10], page_number=1)
    assert len(mapped) == 1
    bbox = mapped[0].get("bbox")
    # union of word_boxes[2:5] => [min x1, min y1, max x2, max y2]
    assert bbox == [101, 0, 220, 10]
