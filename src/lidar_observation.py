"""Keep raw obstacle provenance. This summary does not mask any return."""
import math

def summarize(scan):
    points=[(i,float(r)) for i,r in enumerate(scan.ranges)
            if math.isfinite(r) and scan.range_min<r<scan.range_max]
    nearest=min(points,key=lambda p:p[1]) if points else None
    close=[dict(index=i,range_m=r,bearing_rad=scan.angle_min+i*scan.angle_increment)
           for i,r in points if r<.30]
    return dict(frame=scan.header.frame_id,valid=len(points),
                nearest=nearest[1] if nearest else None,
                nearest_bearing_rad=scan.angle_min+nearest[0]*scan.angle_increment if nearest else None,
                close_count=len(close),close_points=close[:16],mask_applied=False)
