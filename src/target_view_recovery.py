"""Finite active inspection; no stale coordinates or contact permission."""

FLOOR_VIEWS = ("down", "down_near", "down_left", "down_right",
               "down_near_left", "down_near_right")


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
