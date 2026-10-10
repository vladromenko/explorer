import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import test_factory_delivery as fixtures
from factory_delivery import SCOPES, SCOPE_ARTIFACTS, exercise_source, load_factory_settings, record_exercise, seal_factory_exercises


class FactoryExerciseTests(unittest.TestCase):
    def source(self, root, evidence, scope):
        original = root/("data/"+scope+".json")
        row=json.loads(original.read_text())
        row["kind"]=scope+"_exercise"
        row["artifacts"]={key:evidence["artifacts"][key] for key in SCOPE_ARTIFACTS[scope]}
        if scope=="factory_arm":
            row.update(start_deg=[90]*6,goal_deg=[92]*6,boot_id="boot",trajectory_sha256="a"*64,
                       command_completed=True,publish_count=3,cancel_no_further_publications=True)
        elif scope=="camera_geometry":
            row["comparisons"]=[{"translation_error_m":.003,"rotation_error_deg":1.},
                                {"translation_error_m":.004,"rotation_error_deg":1.2}]
        elif scope=="localization":
            row.update(lidar_median_absolute_error_m=[.01,.02],translation_error_m=.02,yaw_error_deg=1.,places_checked=["sock","basket"])
        else:
            def frame(stamp,object_z,tcp_z,clearance):
                return {"at":stamp,"object_id":"sock-1","confidence":.95,"depth_validated":True,
                        "identity_association_verified":True,"frame":"base_footprint","camera_pose_measured":False,
                        "camera_pose_source":"command_estimate","camera_pose_validated_for_execution":True,
                        "camera_pose_validation_record":"data/camera-validation.json","object_xyz":[.3,0,object_z],
                        "tcp_xyz":[.3,0,tcp_z],"floor_clearance_m":clearance,"base_xyyaw":[0,0,0]}
            row.update(open_aperture_mm=50.,closed_gap_mm=5.,guarded_contact_observed=True,
                       before=[frame(row["at"]-4+t,.03,.07,0) for t in (0,.5,1)],after=[frame(row["at"]-4+t,.09,.13,.06) for t in (3,3.5,4)])
        path=root/("data/source-"+scope+".json")
        path.write_text(json.dumps(row))
        return path.relative_to(root).as_posix()

    def fixture(self, directory):
        return fixtures.FactoryDeliveryTests().fixture(directory)

    def test_save_substantive_annotation_without_acceptance_or_movement(self):
        with tempfile.TemporaryDirectory() as directory:
            root,evidence=self.fixture(directory)
            (root/"config/factory-delivery-acceptance.json").unlink()
            source=self.source(root,evidence,"factory_arm")
            result=record_exercise(root,"factory_arm",source,["data/observed.json"],True,"passed")
            self.assertTrue((root/result["path"]).is_file())
            self.assertFalse((root/"config/factory-delivery-acceptance.json").exists())
            self.assertEqual(result["record"]["joint_state_source"],"command_estimate")
            with self.assertRaises(ValueError):
                seal_factory_exercises(root,[result["path"]],True)
            self.assertFalse((root/"config/factory-delivery-acceptance.json").exists())

    def test_unbound_old_source_or_flags_only_cannot_create_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root,evidence=self.fixture(directory)
            source=self.source(root,evidence,"factory_arm")
            path=root/source;row=json.loads(path.read_text());row["artifacts"]={};path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):record_exercise(root,"factory_arm",source,["data/observed.json"],True,"passed")
            row.update(artifacts=evidence["artifacts"]);row.pop("goal_deg");path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):record_exercise(root,"factory_arm",source,["data/observed.json"],True,"passed")
            self.assertFalse((root/"data/factory-delivery-exercises").exists())

    def test_explicit_observation_and_bound_original_images_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root,evidence=self.fixture(directory);source=self.source(root,evidence,"factory_arm")
            with self.assertRaises(ValueError):record_exercise(root,"factory_arm",source,["data/observed.json"],False,"passed")
            (root/"data/other.json").write_text("{}")
            with self.assertRaises(ValueError):record_exercise(root,"factory_arm",source,["data/other.json"],True,"passed")

    def test_failed_operator_result_preserved_but_cannot_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            root,evidence=self.fixture(directory);source=self.source(root,evidence,"factory_arm")
            result=record_exercise(root,"factory_arm",source,["data/observed.json"],True,"failed")
            self.assertEqual(result["record"]["outcome"],"failed")
            with self.assertRaises(ValueError):seal_factory_exercises(root,[result["path"]],True)

    def test_seal_dry_run_no_config_write_and_write_only_all_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            root,evidence=self.fixture(directory);(root/"config/factory-delivery-acceptance.json").unlink()
            names=[]
            # Synthetic geometry runs the real lift verifier; these are temporary
            # test contracts, never physical records for the Explorer installation.
            for scope in sorted(SCOPES):
                source=self.source(root,evidence,scope)
                names.append(record_exercise(root,scope,source,["data/observed.json"],True,"passed")["path"])
            check=seal_factory_exercises(root,names,False)
            self.assertFalse(check["saved"])
            self.assertFalse((root/"config/factory-delivery-acceptance.json").exists())
            written=seal_factory_exercises(root,names,True)
            self.assertTrue(written["saved"])
            self.assertFalse(load_factory_settings(root)["physical_delivery_verified"])
            self.assertEqual(written["evidence"]["physical_test_records"].keys(),dict.fromkeys(names).keys())

    def test_real_gripper_verifier_rejects_empty_frames(self):
        source={"kind":"gripper_contact_exercise","hardware_executed":True,"simulation":False,"outcome":"passed",
                "open_aperture_mm":50,"closed_gap_mm":5,"guarded_contact_observed":True,"before":[],"after":[]}
        with self.assertRaises(ValueError):exercise_source("gripper_contact",source)

    def test_cli_status_reports_specific_missing_artifact_without_write(self):
        script=Path(__file__).resolve().parents[1]/"bin/accept-factory-exercise.py"
        spec=importlib.util.spec_from_file_location("factory_exercise_cli",script)
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as directory,patch("builtins.print") as output:
            result=cli.main(["--root",directory,"status"])
            self.assertEqual(result,0)
            self.assertIn("factory-delivery-acceptance.json",output.call_args.args[0])
            self.assertFalse((Path(directory)/"config").exists())


if __name__=="__main__":unittest.main()
