"""Image/depth geometry hypotheses. No IK targets, actuation or force claims."""
import math
import numpy as np


def candidates(depth, k, bbox, now, image_stamp, frame_stamp):
    if not 0<=now-frame_stamp<1.5 or abs(image_stamp-frame_stamp)>.12:
        raise ValueError('Кадр цели и глубина устарели или не синхронизированы')
    values=np.asarray(bbox,dtype=float)
    if values.shape!=(4,) or not np.isfinite(values).all():
        raise ValueError('Нужна прямоугольная область цели')
    h,w=depth.shape
    x1,y1,x2,y2=map(int,values)
    if not (0<=x1<x2<=w and 0<=y1<y2<=h):
        raise ValueError('Область вне изображения')
    fx,fy=float(k[0,0]),float(k[1,1])
    if not fx>0 or not fy>0:raise ValueError('Нет калибровки камеры')
    roi=depth[y1:y2,x1:x2]
    valid=roi[np.isfinite(roi)&(roi>.15)&(roi<1.5)]
    if valid.size<20:raise ValueError('Недостаточно достоверной глубины в области')
    z=float(np.median(valid));spread=float(np.percentile(valid,90)-np.percentile(valid,10))
    center=[(x1+x2)/2,(y1+y2)/2]
    xyz=[(center[0]-k[0,2])*z/fx,(center[1]-k[1,2])*z/fy,z]
    widths=[(x2-x1)*z/fx,(y2-y1)*z/fy]
    result=[]
    for angle,width in zip((0,90),widths):
        reasons=['camera→base не подтверждён','раскрытие захвата не откалибровано','IK/подход ещё не проверены']
        if spread>.04:reasons.append('Глубина области неоднородна: bbox не является маской предмета')
        result.append(dict(roll_in_image_deg=angle,bbox_extent_m=round(width,4),
            score=round(1/(1+width+5*spread),4),center_camera_m=[float(v) for v in xyz],
            feasible=None,blocked_by=reasons,source='rgbd_bbox_geometry',mask_verified=False,
            force_newtons=None,servo_target=None))
    result.sort(key=lambda r:r['score'],reverse=True)
    return dict(candidates=result,depth_spread_m=spread,valid_depth_pixels=int(valid.size),
                error_to_image_center_px=[float(center[0]-k[0,2]),float(center[1]-k[1,2])],
                summary='Две геометрические гипотезы ранжированы по ширине и неоднородности глубины. Допуска к захвату нет.')
