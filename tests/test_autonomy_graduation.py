import json,tempfile
from pathlib import Path
from autonomy_graduation import AutonomyGraduation

CONFIG=Path(__file__).parents[1]/'config/autonomy-graduation.json'

def setup():
    temp=tempfile.TemporaryDirectory();root=Path(temp.name)
    (root/'config').mkdir();(root/'data/mobile-demonstrations').mkdir(parents=True);(root/'data/learning-jobs').mkdir()
    (root/'config/autonomy-graduation.json').write_bytes(CONFIG.read_bytes())
    (root/'config/commissioning.json').write_text('{}')
    return temp,root,AutonomyGraduation(root)

def evidence(root,n,kind,outcome='success',**extra):
    row=dict(schema='explorer_autonomy_evidence_v1',kind=kind,hardware_executed=True,simulation=False,verifier_outcome=outcome,**extra)
    (root/'data/autonomy-evidence'/f'{n}.json').write_text(json.dumps(row))

def test_operator_labels_alone_do_not_unlock_grasp():
    temp,root,g=setup()
    try:
        for n in range(30):
            p=root/'data/mobile-demonstrations'/str(n);p.mkdir();(p/'episode.json').write_text(json.dumps(dict(state='complete',outcome='success',stages=['travel_to_object','grasp','carry','place'])))
        assert g.status()['items'][1]['state']=='training'
        assert json.loads((root/'config/commissioning.json').read_text())['gripper_calibrated'] is False
    finally:temp.cleanup()

def test_verified_localization_promotes_and_revokes():
    temp,root,g=setup()
    try:
        for n,h in enumerate(('front','left','right','front','left','right')):
            evidence(root,n,'localization_return',translation_error_m=.04,yaw_error_deg=2,start_heading_bucket=h)
        assert g.status()['items'][0]['state']=='accepted'
        assert json.loads((root/'config/commissioning.json').read_text())['localization_verified'] is True
        for p in (root/'data/autonomy-evidence').glob('*.json'):p.unlink()
        assert g.status()['items'][0]['state']=='training'
        assert json.loads((root/'config/commissioning.json').read_text())['localization_verified'] is False
    finally:temp.cleanup()

def test_grasp_needs_machine_evidence_and_writes_bounded_calibration():
    temp,root,g=setup()
    try:
        evidence(root,'aperture','gripper_aperture_calibration',open_deg=40,sock_close_deg=150,open_aperture_mm=55)
        for n in range(30):
            p=root/'data/mobile-demonstrations'/str(n);p.mkdir()
            (p/'episode.json').write_text(json.dumps(dict(state='complete',outcome='failure' if n<5 else 'success',stages=['travel_to_object','grasp','carry','place'])))
        for n in range(15):evidence(root,'lift'+str(n),'retained_lift',outcome='failure' if n<3 else 'success',object_class='soft_cloth',condition=('flat','folded','dark')[n%3])
        for n in range(5):evidence(root,'place'+str(n),'verified_place')
        assert g.status()['items'][1]['state']=='accepted'
        accepted=json.loads((root/'config/gripper-accepted.json').read_text())
        assert accepted['execution_authorized'] is True and accepted['feedback']=='visual_closed_loop'
    finally:temp.cleanup()
