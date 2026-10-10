"""Suppress duplicate boxes, retaining separate targets and the raw hypotheses."""
import math


def unique_candidates(rows, iou_threshold=0.5):
    if not 0 < iou_threshold < 1:
        raise ValueError("Invalid IoU threshold")
    for row in rows:
        box = row.get("bbox")
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
                or box[2] <= box[0] or box[3] <= box[1]
                or type(row.get("confidence")) not in (int, float)
                or not math.isfinite(row["confidence"])):
            raise ValueError("Invalid detection candidate")
    selected = []
    for row in sorted(rows, key=lambda item: item["confidence"], reverse=True):
        a = row["bbox"]
        duplicate = False
        for previous in selected:
            if row.get("label") == previous.get("label"):
                b = previous["bbox"]
                intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
                union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
                if intersection / union > iou_threshold:
                    duplicate = True
                    break
        if not duplicate:
            selected.append(row)
    return selected
