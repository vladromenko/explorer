import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from controller_release import atomic_json,digest,select_controller_profile,stage_release


class ControllerReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);self.release=self.root/'release'
        (self.root/'config').mkdir();(self.root/'data').mkdir();self.release.mkdir()
        self.board=dict(uid='12'*12,device=0x450,revision=0x2003,flash_kib=1024)
        calibration=self.root/'config/controller-calibration.json';calibration.write_text('calibration fixture')
        self.profile=dict(transport='controller_v1',device='/dev/fixture',firmware_source_sha256='a'*64,
            calibration_sha256=digest(calibration),telemetry_only=False,hardware_accepted=True)
        atomic_json(self.root/'config/controller-profile.json',self.profile)
        self.current=dict(self.board,source_sha256='a'*64,boot=1)
        atomic_json(self.root/'data/controller-state.json',dict(at=time.time(),identity=self.current,telemetry_fresh=True))
        self.source='b'*64;application=b'application fixture'+bytes.fromhex(self.source)
        image=application+bytes(131072-len(application));self.image_sha=hashlib.sha256(image).hexdigest()
        files={'explorer-sector0.bin':image,'application/explorer.bin':application,'recovery/installed-sector0.bin':bytes(131072)}
        for name,value in files.items():
            p=self.release/name;p.parent.mkdir(exist_ok=True);p.write_bytes(value)
        self.manifest=dict(schema=2,version='test-release',image_sha256=self.image_sha,
            source_sha256=self.source,sector_bytes=131072,first_sector_address='0x8000000',measured_identity=self.current,
            files={name:hashlib.sha256(value).hexdigest() for name,value in files.items()})
        atomic_json(self.release/'manifest.json',self.manifest)

    def stage(self):return stage_release(self.root,self.release,self.image_sha)

    def test_staging_changes_no_active_profile_and_known_hello_switches_to_telemetry(self):
        result=self.stage();self.assertFalse(result['active_profile_changed']);self.assertFalse(result['motion_enabled'])
        self.assertEqual(json.loads((self.root/'config/controller-profile.json').read_text()),self.profile)
        identity=dict(self.current,source_sha256=self.source,boot=2)
        selected=select_controller_profile(self.root,self.profile,identity)
        self.assertTrue(selected['telemetry_only']);self.assertFalse(selected['hardware_accepted'])
        self.assertEqual(selected['device'],'/dev/fixture');self.assertEqual(selected['calibration_sha256'],self.profile['calibration_sha256'])
        self.assertEqual(select_controller_profile(self.root,selected,identity),selected)
        self.assertEqual(len(list((self.root/'data/controller-profile-history').glob('*.json'))),1)
        restored=select_controller_profile(self.root,selected,self.current)
        self.assertEqual(restored['firmware_source_sha256'],'a'*64)
        self.assertTrue(restored['telemetry_only']);self.assertFalse(restored['hardware_accepted'])

    def test_unknown_image_or_foreign_board_never_changes_profile(self):
        self.stage()
        for identity in (dict(self.current,source_sha256='c'*64),dict(self.current,source_sha256=self.source,uid='34'*12),
                         dict(self.current,uid='34'*12),dict(self.current,source_sha256=self.source,flash_kib=2048)):
            with self.assertRaises(ValueError):select_controller_profile(self.root,self.profile,identity)
        self.assertEqual(json.loads((self.root/'config/controller-profile.json').read_text()),self.profile)

    def test_modified_calibration_and_concurrent_profile_are_not_overwritten(self):
        self.stage();identity=dict(self.current,source_sha256=self.source)
        path=self.root/'config/controller-profile.json';changed=dict(self.profile,device='/dev/changed');atomic_json(path,changed)
        with self.assertRaises(ValueError):select_controller_profile(self.root,self.profile,identity)
        self.assertEqual(json.loads(path.read_text()),changed)
        atomic_json(path,self.profile);(self.root/'config/controller-calibration.json').write_text('changed')
        with self.assertRaises(ValueError):select_controller_profile(self.root,self.profile,identity)

    def test_staging_checks_hashes_fresh_identity_and_calibration(self):
        with self.assertRaises(ValueError):stage_release(self.root,self.release,'f'*64)
        p=self.root/'data/controller-state.json'
        atomic_json(p,dict(at=time.time()-10,identity=self.current,telemetry_fresh=True))
        with self.assertRaises(ValueError):self.stage()
        atomic_json(p,dict(at=time.time(),identity=self.current,telemetry_fresh=True))
        (self.release/'application/explorer.bin').write_bytes(b'bad')
        with self.assertRaises(ValueError):self.stage()

    def test_package_path_cannot_escape_release(self):
        outside=self.root/'outside';outside.write_text('outside')
        manifest=copy.deepcopy(self.manifest);manifest['files']['../outside']=digest(outside)
        atomic_json(self.release/'manifest.json',manifest)
        with self.assertRaises(ValueError):self.stage()

if __name__=='__main__':unittest.main()
