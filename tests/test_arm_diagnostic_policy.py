"""Exercise diagnostic command policy without opening any hardware device."""
import ast
from dataclasses import asdict
import hashlib
import importlib.util
import io
import json
import tempfile
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'bin/check-controller-arm.py'


class DiagnosticPolicyTests(unittest.TestCase):
    def load_script(self):
        spec=importlib.util.spec_from_file_location('arm_diagnostic',SCRIPT)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_zero_and_relief_without_execute_never_open_uart(self):
        module=self.load_script()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'data').mkdir()
            state=root/'data/controller-state.json';state.write_text('{}')
            module.ROOT=root
            with patch('sys.argv',[str(SCRIPT),'0','--gripper-relief']),patch('sys.stdout',new_callable=io.StringIO) as out:
                module.main()
            report=json.loads(out.getvalue())
            self.assertFalse(report['uart_opened']);self.assertFalse(report['commands_sent'])
            self.assertIsNone(module.s)
            self.assertEqual(state.read_text(),'{}')

    def test_current_outside_pose_cannot_invent_recovery_calibration(self):
        module=self.load_script()
        from controller_feedback import vendor_calibration
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'config').mkdir()
            cal=root/'config/controller-calibration.json'
            cal.write_text(json.dumps(dict(schema=1,joints=[asdict(c) for c in vendor_calibration()])))
            before=cal.read_bytes();digest=hashlib.sha256(before).hexdigest()
            (root/'config/controller-profile.json').write_text(json.dumps(dict(calibration_sha256=digest,firmware_source_sha256='a'*64)))
            loaded=module.approved_calibration(root,dict(identity=dict(source_sha256='a'*64)))
            samples=[dict(joint=i+1,raw_ticks=2000,fresh=True,raw_valid=True,position_valid=True,error=0,device_error=0) for i in range(6)]
            samples[2]['raw_ticks']=3576
            with self.assertRaisesRegex(ValueError,'Joint 3.*no verified recovery corridor'):
                module.checked_bounds(loaded,samples)
            self.assertEqual(cal.read_bytes(),before)
            self.assertEqual(loaded[2].recovery_min,0)

    def command_environment(self, replies, partial=False):
        # Loading the script itself opens serial, so compile only its dispatcher.
        tree = ast.parse(SCRIPT.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'cmd')
        module = ast.Module(body=[function], type_ignores=[])
        clock = [1.0]
        writes, preparations, results = [], [], []
        session = SimpleNamespace(sequence=0)

        def prepare(*args, **kwargs):
            preparations.append((args, kwargs))
            session.sequence += 1
            return b'packet'

        def write(packet):
            writes.append(packet)
            return len(packet) - int(partial)

        def poll(seconds):
            clock[0] += seconds
            if replies:
                results.append(replies.pop(0))

        session.prepare = prepare
        env = dict(phase_four=False, results=results, session=session,
                   events=[], s=SimpleNamespace(write=write), poll=poll,
                   time=SimpleNamespace(monotonic=lambda: clock[0], monotonic_ns=lambda: int(clock[0]*1e9)))
        exec(compile(module, str(SCRIPT), 'exec'), env)
        return env['cmd'], writes, preparations

    def test_rejection_never_renews_or_retries(self):
        command, writes, prepared = self.command_environment([dict(op=15, seq=1, result=16)])
        with self.assertRaises(RuntimeError):
            command(15)
        self.assertEqual(len(writes), 1)
        self.assertEqual(len(prepared), 1)

    def test_unrelated_success_is_not_our_ack(self):
        command, writes, prepared = self.command_environment([dict(op=15, seq=999, result=0)])
        with self.assertRaisesRegex(RuntimeError, 'No matching'):
            command(15)
        self.assertEqual(len(writes), 1)
        self.assertEqual(len(prepared), 1)

    def test_matching_ack_and_partial_write(self):
        command, writes, _ = self.command_environment([dict(op=15, seq=1, result=0)])
        command(15)
        self.assertEqual(len(writes), 1)
        command, writes, _ = self.command_environment([], partial=True)
        with self.assertRaisesRegex(RuntimeError, 'Partial'):
            command(15)
        self.assertEqual(len(writes), 1)


if __name__ == '__main__':
    unittest.main()
