"""Copy explicitly allowlisted Git sources and nonsecret experiment evidence."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'docs/independent_learning_audit'
REVISION = '1da8ecb26efaa798d4293f126d7d123aff7985e5'


def main():
    tree = ast.parse((ROOT/'simulator/room_policy.py').read_text())
    files = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='RUNTIME_FILES' for t in node.targets):
            files.update(ast.literal_eval(node.value))
    files.update(['src/learning_stack.py','src/local_grasp_experiment.py','src/lerobot_bridge.py',
        'src/mobile_demonstrations.py','src/mobile_policy_execution.py','src/skill_learning.py',
        'src/skill_planner.py','src/arm_planner.py','src/arm_planner_worker.py','src/visual_navigation.py','src/autonomy_runtime.py','src/learning_workflows.py',
        'src/skill_discovery.py','src/factory_delivery.py','src/factory_trajectory.py',
        'simulator/surface_destination.py','docs/LEARNING-IMPLEMENTATION.ru.md',
        'src/controller_driver.py','src/grasp_verification.py','src/delivery_vision.py',
        'src/model_backends.py','simulator/skill_training.py','simulator/automatic_learning.py',
        'simulator/language_learning.py','simulator/phase_learning.py',
        'simulator/scan_navigation.py','simulator/manipulation_planning.py',
        'bin/queue-room-cleanup.py','bin/resume-paused-learning.py','bin/diagnose-live-room.py',
        'bin/audit-trajectory-similarity.py','bin/benchmark-horizon-reader.py',
        'bin/check-policy-cadence.py','bin/probe-saved-policy.py','bin/probe-room-intervention.py',
        'config/handeye-accepted.json','config/gripper-reference.json','config/controller-calibration.json',
        'config/lidar_geometry.json','config/simulator-macos.lock.txt','config/learning-environment.lock.txt',
        'config/local-grasp-profile.json','docs/firmware-rc2-withdrawal-20260929.md',
        'docs/ARM-RESTORATION-20260929.ru.md','docs/FACTORY-INTEGRATION-20260929.ru.md',
        'docs/EXPLORER-ARM-CURRENT-2026-10-08.ru.md','docs/CODE-AUDIT-20261008.ru.md',
        'docs/calibration-progress-20260928.md','docs/WORKLOG.md'])
    for path in (ROOT/'tests').glob('test_*.py'):
        if any(s in path.name for s in ['room_','learning','horizon','cadence','policy_input','neural_recovery','checkpoint_monitor','live_room','skill_training','failure_repair']):
            files.add(str(path.relative_to(ROOT)))
    files.update(str(p.relative_to(ROOT)) for p in (ROOT/'docs/reports').glob('*.json'))
    manifest, functions, errors = [], [], []
    for name in sorted(files):
        result = subprocess.run(['git','show',REVISION+':'+name],cwd=ROOT,capture_output=True)
        if result.returncode:
            errors.append(dict(path=name,reason='Not present at pinned analysis revision'))
        else:
            target=OUT/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(result.stdout)
            manifest.append(dict(path=name,bytes=len(result.stdout),sha256=hashlib.sha256(result.stdout).hexdigest(),revision=REVISION))
            if name.endswith('.py'):
                try:
                    parsed=ast.parse(result.stdout.decode())
                    for node in ast.walk(parsed):
                        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                            functions.append(dict(path=name,name=node.name,first_line=node.lineno,last_line=node.end_lineno))
                except (SyntaxError,UnicodeError) as exc:
                    errors.append(dict(path=name,reason=str(exc)))
    (OUT/'evidence/source_manifest.json').write_text(json.dumps(dict(revision=REVISION,files=manifest,errors=errors),ensure_ascii=False,indent=2)+'\n')
    lines=['# Функции и классы в закреплённом исходном коде','',f'Ревизия `{REVISION}`. Наличие функции не является экспериментальным успехом.','', '| Файл | Функция / класс | Строки |','| --- | --- | --- |']
    for row in functions:
        url=f"https://github.com/vladromenko/explorer-learning/blob/{REVISION}/{row['path']}#L{row['first_line']}-L{row['last_line']}"
        lines.append(f"| [{row['path']}]({url}) | `{row['name']}` | {row['first_line']}–{row['last_line']} |")
    (OUT/'evidence/SOURCE_FUNCTION_INDEX.md').write_text('\n'.join(lines)+'\n')
    tracked=subprocess.check_output(['git','ls-tree','-r','--long',REVISION],cwd=ROOT,text=True)
    (OUT/'evidence/full_git_tree.txt').write_text(tracked)
    selected=['data-runtime-02.json','checkpoint-001000-proof.json','checkpoint-007000-proof.json',
        'component-benchmark-02.json','geometry-provenance.json','recording-sample-audit.json',
        'trajectory-similarity.json','projected-reader-benchmark.json','policy-probe-001000-1200calls.json',
        'cadence-physics-regression.json','projected-resume-proof.json','room-e673-actor-evidence.json']
    external=Path('/Volumes/ExplorerSim/ExplorerSim/diagnostics/audit-20261019')
    for name in selected:
        if (external/name).is_file():
            dest=OUT/'evidence/historical'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(external/name,dest)
    print(json.dumps(dict(sources=len(manifest),function_index_entries=len(functions),unavailable=errors)))


if __name__=='__main__':
    main()
