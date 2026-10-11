"""Finite active inspection; no stale coordinates or contact permission."""
from detection_candidates import unique_candidates

FLOOR_VIEWS = ("down", "down_near", "down_left", "down_right",
               "down_near_left", "down_near_right")


def visible_search_target(result, width, height):
    """Inspect raw boxes: NMS may discard the enclosing cloth hypothesis."""
    raw = "raw_objects" in result
    objects = result["raw_objects"] if raw else result.get("objects", [])
    return dict(visible_target(objects, width, height),
                inspection_source="raw_hypotheses" if raw else "filtered_hypotheses")


def visible_target(objects, width, height):
    """Accept one enclosing depth hypothesis for inspection, never item count.

    Grounding may label both a sock and its toe. Require all same-label boxes
    to be almost entirely inside the largest box, which must itself have
    resolved foreground depth and be fully visible. Adjacent objects and a
    clipped enclosing box remain ambiguous. Raw hypotheses are preserved.
    """
    unique_candidates(objects)  # Validate all boxes, including rejected ones.
    if not objects:
        return {"found": False, "reason": "target_missing"}
    def area(item):
        box = item["bbox"]
        return (box[2]-box[0])*(box[3]-box[1])
    target = max(objects, key=area)
    a = target["bbox"]
    contained = True
    for item in objects:
        b = item["bbox"]
        overlap = max(0, min(a[2],b[2])-max(a[0],b[0]))*max(0, min(a[3],b[3])-max(a[1],b[1]))
        if item.get("label") != target.get("label") or overlap/area(item) < .97:
            contained = False
    visible = a[0] >= 3 and a[1] >= 3 and a[2] <= width-3 and a[3] <= height-3
    found = contained and visible and bool(target.get("position")) and target["confidence"] >= .35
    return {"found": found, "reason": None if found else "missing_clipped_or_ambiguous_target",
            "target": target if found else None, "nested_detection_group": len(objects)>1 and contained,
            "physical_object_count_verified": False, "identity_verified": False}


def recover_view(observe, move_view, retreat, permit, record, max_retreats=2):
    """Inspect fresh views, then back off 6 cm only if none is usable.

    Callback failures and cancellation propagate. A detector hypothesis is
    never converted into verified identity, contact or joint measurement.
    """
    if type(max_retreats) is not int or not 0 <= max_retreats <= 2:
        raise ValueError("Invalid inspection retreat budget")
    for attempt in range(max_retreats + 1):
        for view in FLOOR_VIEWS:
            permit()
            move_view(view)
            permit()
            result = observe()
            permit()
            record("target_view", {"view": view, "retreat_count": attempt, "observation": result})
            if result.get("found") is True:
                return dict(result, view=view, retreat_count=attempt,
                            contact_authorized=False, identity_verified=False)
        if attempt < max_retreats:
            permit()
            result = retreat(.06)
            permit()
            record("view_retreat", {"retreat_count": attempt + 1, "result": result})
    return {"found": False, "reason": "no_usable_target_after_inspection",
            "retreat_count": max_retreats, "contact_authorized": False}
