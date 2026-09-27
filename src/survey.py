"""Evidence captured at visited waypoints, without inventing object map positions."""
import json
import time
import shutil
from pathlib import Path
from lerobot_bridge import write_json


def summarize(perception,place):
    labels=sorted({o['label'] for o in perception.get('objects',[]) if o.get('confidence',0)>=.6})
    return 'В точке «'+place+'» детектор предполагает: '+(', '.join(labels) if labels else 'нет уверенных распознаваний')+'.'


class SurveyStore:
    def __init__(self,root):
        self.root=Path(root);self.folder=self.root/'data/surveys';self.folder.mkdir(exist_ok=True)

    def capture(self,mission,place,pose,map_epoch,arrived_at,permit):
        deadline=time.monotonic()+5
        perception=None
        while time.monotonic()<deadline:
            permit()
            candidate=json.loads((self.root/'data/perception.json').read_text())
            stamp=candidate.get('image_stamp',0)
            if arrived_at<stamp<=time.time() and time.time()-stamp<2:
                perception=candidate;break
            time.sleep(.05)
        if perception is None:raise ValueError('Нет нового кадра после прибытия')
        permit()
        state=json.loads((self.root/'data/status.json').read_text())
        now=time.time()
        if not 0<=now-state.get('at',0)<1:raise ValueError('Состояние датчиков устарело')
        folder=self.folder/mission;folder.mkdir(exist_ok=True)
        ident=str(time.time_ns())
        record=dict(id=ident,mission=mission,at=now,place=place,robot_pose=pose,map_epoch=map_epoch,
                    perception=perception,lidar=state.get('lidar'),battery_voltage_v=state.get('battery'),
                    object_map_positions_verified=False,summary=summarize(perception,place))
        # The JPEG is supporting context; its exact frame identity is not asserted.
        image=self.root/'data/frame-raw.jpg'
        if image.exists() and 0<=now-image.stat().st_mtime<2:
            shutil.copyfile(image,folder/(ident+'.jpg'))
            record['context_image']=ident+'.jpg'
            record['context_image_same_frame_verified']=False
        write_json(folder/(ident+'.json'),record)
        return record

    def recent(self):
        paths=sorted(self.folder.glob('*/*.json'),key=lambda p:p.name,reverse=True)[:30]
        return [json.loads(p.read_text()) for p in paths]
