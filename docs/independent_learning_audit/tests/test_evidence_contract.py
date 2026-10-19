"""Cross-check independent saved sources; no model, MuJoCo or robot imports."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text())


class EvidenceContract(unittest.TestCase):
    def test_pinned_source_bytes(self):
        manifest = read('evidence/source_manifest.json')
        self.assertFalse(manifest['errors'])
        for row in manifest['files']:
            with self.subTest(path=row['path']):
                self.assertEqual(hashlib.sha256((ROOT / 'source' / row['path']).read_bytes()).hexdigest(), row['sha256'])

    def test_refusal_matches_original_result_and_trace(self):
        result = read('evidence/refusal-original/trial/result.json')
        forensic = read('examples/refusal-695/refusal_forensics.json')
        refusal = result['policy_command_refusal']
        self.assertEqual(refusal['step'], 695)
        self.assertFalse(refusal['decision']['queued'])
        self.assertEqual(refusal['decision']['prediction'], forensic['recorded_refusal']['decision']['prediction'])
        delta = refusal['decision']['prediction'][5] - refusal['decision']['reference'][5]
        self.assertGreater(delta, 20.)
        self.assertAlmostEqual(delta, 27.8205337524414)
        self.assertEqual(hashlib.sha256((ROOT/'examples/refusal-695/actions.jsonl').read_bytes()).hexdigest(),
                         forensic['trace_sha256'])
        self.assertTrue(forensic['replay_matches_atol_1e_4'])
        self.assertFalse(forensic['guard']['accepted'])
        self.assertEqual(len(forensic['full_denormalized_chunk'][0]), 20)
        self.assertTrue(all(len(row) == 9 for row in forensic['full_denormalized_chunk'][0]))
        self.assertFalse(result['success'])
        self.assertTrue(result['collision_free'])
        self.assertFalse(result['physical_outcome']['lifted'])

    def test_runtime_outside_measured_training_range(self):
        stats = read('evidence/runtime_distribution_followup.json')
        result = read('evidence/refusal-original/trial/result.json')
        shoulder = result['policy_command_refusal']['decision']['reference'][1]
        self.assertGreater(shoulder, stats['valid_observation_max'][1])
        self.assertEqual(stats['valid_observations_shoulder_ge_123'], 0)
        self.assertEqual(stats['training_phases_at_40_75_s'],
                         {'release_retreat': 728, 'release_unhook': 6, 'place': 1})

    def test_dataset_count_reconciles_recordings(self):
        stats = read('dataset_statistics.json')
        recordings = read('evidence/recording_statistics.json')
        self.assertEqual(len(recordings), 739)
        self.assertEqual(sum(r['frames'] for r in recordings), stats['total_raw_rows'])
        self.assertEqual(sum(r['masked_frames'] for r in recordings), stats['target_masked_rows'])
        self.assertEqual(sum(r['split'] == 'train' for r in recordings), 590)
        self.assertEqual(sum(r['split'] == 'validation' for r in recordings), 149)

    def test_pause_and_no_audit_actuation(self):
        snapshot = read('system_snapshot.json')
        states = snapshot['states']
        self.assertEqual(states['neural-recovery/924d1ee5ab01/state.json']['value']['state'], 'paused')
        self.assertEqual(states['room-learning/0c946789dbd7/state.json']['value']['state'], 'operator_paused')
        self.assertEqual(states['runs/d4f59a63085b/job.json']['value']['steps_completed'], 29078)
        actions = snapshot['audit_actions']
        self.assertEqual(actions['training_updates'], 0)
        self.assertEqual(actions['commands_to_robot'], 0)
        self.assertFalse(actions['queues_resumed'])


if __name__ == '__main__':
    unittest.main()
