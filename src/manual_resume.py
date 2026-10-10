"""Confirm core state before granting a manual arm input lease."""
import time


def resume_confirmed(emit, read, timeout=1.5, clock=time.monotonic, wait=time.sleep):
    emit("clear_stop")
    request = emit("mode", mode="MANUAL")
    deadline = clock()+timeout
    while clock() < deadline:
        state = read()
        if (state.get("last_request", {}).get("id") == request["id"]
                and state.get("mode") == "MANUAL" and state.get("stop_latched") is False
                and 0 <= time.time()-state.get("at", 0) < 1):
            return {"core_confirmed": True, "request_id": request["id"]}
        wait(.02)
    emit("stop", initiator="manual_resume_timeout")
    raise ValueError("Контроллер не подтвердил ручной режим без STOP; движение не начато")
