"""Read-only evidence collection and audit-only document assembly.

Never starts a learner, simulator rollout, ROS interface or actuator.
"""
import csv
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

OUT = Path(__file__).resolve().parents[1]
REPO = OUT.parents[1]
ROOT = Path('/Volumes/ExplorerSim/ExplorerSim')
REV = '1da8ecb26efaa798d4293f126d7d123aff7985e5'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_evidence():
    diag = ROOT / 'diagnostics/live-room-395c6a0365d9'
    paths = [diag / name for name in ['request.json', 'handoff.json', 'state.json',
        'trial/result.json', 'trial/observed/policy-command-refusal.json']]
    paths.extend(p for p in (diag / 'model').rglob('*')
        if p.is_file() and p.name in ['config.json', 'train_config.json', 'bundle.json',
            'policy_preprocessor.json', 'policy_postprocessor.json',
            'policy_preprocessor_step_3_normalizer_processor.safetensors',
            'policy_postprocessor_step_0_unnormalizer_processor.safetensors'])
    rows = []
    for p in paths:
        target = OUT / 'evidence/refusal-original' / p.relative_to(diag)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        rows.append(dict(source=str(p), packet=str(target.relative_to(OUT)), bytes=p.stat().st_size,
                         sha256=sha(p)))
    dump(OUT / 'evidence/selected_artifact_manifest.json', rows)
    versions = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                if d.metadata.get('Name')}
    dump(OUT / 'evidence/dependencies.json', dict(python=sys.version, distribution_versions=versions,
        platform=subprocess.check_output(['uname', '-a'], text=True).strip(),
        environment_python='/Volumes/ExplorerSim/env/bin/python'))
    import inspect
    from lerobot.policies.act.modeling_act import ACTPolicy
    package = OUT / 'source_external'
    package.mkdir(exist_ok=True)
    parts = ['# Installed LeRobot ACTPolicy excerpts; source locations and SHA below.\n']
    for name in ['reset', 'select_action', 'predict_action_chunk']:
        method = getattr(ACTPolicy, name)
        parts.append(f'\n# ACTPolicy.{name}, original line {inspect.getsourcelines(method)[1]}\n')
        parts.append(inspect.getsource(method))
    (package / 'lerobot_act_policy_methods.py').write_text(''.join(parts))
    original = Path(inspect.getsourcefile(ACTPolicy))
    dump(package / 'provenance.json', dict(distribution=versions.get('lerobot'),
        original_file=str(original), original_sha256=sha(original),
        purpose='Library queue behaviour; excerpts only, not a standalone importable replacement'))


KNOWN = [
('firmware_sync_write', '2026-09-29', 'historical firmware', 'Wrong assertion that factory sync-write was absent',
 'Arm_Sync_Write existed; rc2 withdrawn before flashing', 'VERIFIED', 'source/docs/firmware-rc2-withdrawal-20260929.md', 'withdrawn; no proof of physical fix'),
('pipeline_scope', '2026-09-29/10-04', 'learning UI', 'Ready wording preceded a complete mobile policy executor',
 'Component readiness substituted for end-to-end acceptance', 'STRONGLY_SUPPORTED', '19_CONVERSATION_AND_DECISION_AUDIT.md#d04', 'not qualified'),
('browser_stop', '2026-10-12', 'night cycle', 'Cycle stopped about 46 seconds after leaving tab',
 'Browser focus/visibility stop coupled to background job', 'VERIFIED', '02_FULL_HISTORY.md', '0293b06; background behaviour corrected'),
('exposure_budget', '2026-10-12', 'local ACT', '20000 updates with 129548 frames/batch16 = 2.47 passes',
 'Too few passes relative to configured minimum8', 'VERIFIED', '02_FULL_HISTORY.md', 'minimum exposure corrected; did not establish full success'),
('unpaired_regression', '2026-10-14', 'd6ebe13cae23 / later run', '6/10 then 4/10 with changed scenes/initialization',
 'Attribution to model/data change confounded by evaluation changes', 'VERIFIED', '19_CONVERSATION_AND_DECISION_AUDIT.md#d12', 'paired gates added later'),
('maintenance_pause', '2026-10-15', 'fb56a108b63f', 'Installer maintenance pause interpreted as user pause',
 'Pause origin was not separated', 'VERIFIED', '19_CONVERSATION_AND_DECISION_AUDIT.md#d14', '3fb6905; old cycle now permanently stopped'),
('neural_consumer_absent', '2026-10-15/17', 'hierarchical skill training', 'Parameter trials did not train ACT',
 'Architecture change dropped required dataset-to-neural-model consumer', 'VERIFIED', '19_CONVERSATION_AND_DECISION_AUDIT.md#d15', '2161826/853c4c0 restore neural pipeline'),
('renderer_stall', '2026-10-17', 'skill workers', '167 attempts unchanged for more than2h',
 'Render path blocked worker', 'VERIFIED', '19_CONVERSATION_AND_DECISION_AUDIT.md#d17', 'render execution repaired; no skill acceptance implied'),
('equal_score', '2026-10-18', '42c01dcd50a8-01', '119047 updates, candidate3/10 reference3/10',
 'No measured improvement; exact learning cause UNKNOWN', 'VERIFIED', 'evidence/run_inventory.json', 'reference retained; corrections collected'),
('lineage', '2026-10-19', 'early diagnostic d42fb2154305', 'Interrupted same-ID copy shadowed exact training source',
 'Episode lookup ignored training content hash', 'VERIFIED', 'source/simulator/episode_lineage.py', '3568b30 exact-byte resolution'),
('socket_path', '2026-10-19', 'early private vision', 'Private service UNIX socket path too long',
 'Deep frozen source path', 'VERIFIED', 'source/simulator/live_room_diagnostic.py', '929497a short private vprobe root'),
('frontier', '2026-10-19', '8dae7bbaff62, scene1703000014', 'Sock found, basket not confirmed after24 views',
 'Cheap low-information turns dominated useful views', 'STRONGLY_SUPPORTED', '02_FULL_HISTORY.md', '934c479; later both objects found'),
('resource_wait_budget', '2026-10-19', '62b4807fa805', 'Hour expired before scene start',
 'Resource waiting incorrectly consumed active execution budget', 'VERIFIED', 'source/simulator/live_room_diagnostic.py', '8c6ce11 active-time-only budget'),
('rim_aim', '2026-10-19', 'bb4a5a910436', 'Basket survey wrist pose unreachable',
 'Aimed at sidewall centre rather than framing observed rim', 'STRONGLY_SUPPORTED', 'source/simulator/surface_destination.py', 'c746696; subsequent both-object confirmation'),
('cad_aabb', '2026-10-19', '87b2454a6c81', 'Fresh geometric alignment did not converge before ACT',
 'World AABB empty corner falsely intersected tilted full CAD mesh', 'VERIFIED', '02_FULL_HISTORY.md', 'a5d9ea7 separating-plane refinement; obstacle control still rejected'),
('optical_range', '2026-10-19', 'b3b879bcdeee', 'Four close views refused before ACT',
 'Fixed .34m optical range infeasible at .20m standoff', 'VERIFIED', '02_FULL_HISTORY.md', '726f8cb finite feasible distances with unchanged gates'),
('cadence', '2026-10-19', '8c8e4fbc4b7f', 'Fresh unsafe shoulder/elbow output after200 actions',
 '12 manual ticks after10ACT produced22s physical versus10s actor clock; contribution not isolated', 'VERIFIED', 'evidence/historical/cadence-physics-regression.json', 'cfe4aa7 timing10/10; later grasp still failed'),
('reader_decode', '2026-10-19', 'd4f59a63085b', 'Slow data load/updates',
 'Action horizon decoded20 unused future RGB frames', 'VERIFIED', 'evidence/historical/projected-reader-benchmark.json', '1000 to0 extra decodes in50 queries; targets byte-identical'),
('memory_accounting', '2026-10-19', 'early diagnostic', 'False resource wait after Docker closure',
 'Bypassed native resource_budget; psutil and native availability differed', 'VERIFIED', 'source/simulator/live_room_diagnostic.py', '09cb728 common native budget; thresholds not lowered'),
('incomplete_checkpoint', '2026-10-19', 'd4f59a63085b checkpoint018000', 'Weights existed without complete optimizer state',
 'Disk disconnect interrupted checkpoint save', 'VERIFIED', 'checkpoints.csv', 'safe resume from complete017000; incomplete data preserved'),
('refusal_695', '2026-10-19', '395c6a0365d9 scene1703000014 checkpoint009000', 'Fresh gripper48→75.8205 refused; no grasp/lift/place',
 'Numerical mechanism verified; unique learning cause UNKNOWN', 'VERIFIED', 'examples/refusal-695/refusal_forensics.json', 'read-only replay exact; no repair or new physical rollout claimed'),
('state_time_shift', '2026-10-19 audit', 'd4f59a63085b / same refusal', 'Shoulder123 exceeds training max116.5517; clock40.75 associated with retreat',
 'Confirmed state/phase mismatch; causal proportion UNKNOWN', 'VERIFIED', 'evidence/runtime_distribution_followup.json', 'requires contract/recovery ablations before long training'),
('feedback_claim', '2026-10-19 audit', 'assistant audit interpretation', 'Old raw replies were overinterpreted as current joint feedback',
 'Historical interface confused with current command-only configuration', 'VERIFIED', '19_CONVERSATION_AND_DECISION_AUDIT.md#d26', 'claim withdrawn after user correction'),
]


def failures():
    fields = ['id', 'date', 'run_or_scene', 'symptom', 'cause', 'status', 'evidence', 'resolution']
    rows = [dict(zip(fields, r)) for r in KNOWN]
    raw = read(OUT / 'evidence/raw_episode_inventory.json')
    for item in raw:
        if item.get('success') is False:
            ep = read(Path(item['path']))
            rows.append(dict(id=item['id'], date='UNKNOWN', run_or_scene=str(item.get('seed')),
                symptom=str(ep.get('failure_reason', ep.get('reason', 'Manifest success=false'))),
                cause='UNKNOWN: outcome declaration is not a root-cause diagnosis', status='VERIFIED',
                evidence=item['path'], resolution='Preserved; not reevaluated during audit'))
    for p in sorted((ROOT / 'diagnostics').glob('live-room-*/trial/result.json')):
        d = read(p)
        if not d.get('success', False):
            rows.append(dict(id=p.parents[1].name, date='2026-10-19',
                run_or_scene=str(d.get('seed', 'UNKNOWN')), symptom=str(d.get('reason', d.get('state'))),
                cause='See classified cases if matched; otherwise UNKNOWN', status='VERIFIED',
                evidence=str(p), resolution='Historical result preserved; repeated scene is not final qualification'))
    with (OUT / 'failures.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    text = '''# 08. База найденных отказов и границы классификации

VERIFIED относится к наличию события или декларации результата. Оно не означает, что каждая предполагаемая причина доказана. В `failures.csv` сохранены все найденные `success=false` поддержанных семейств raw episodes, завершённые ранние диагностические отказы и отдельные инженерные случаи. Дубликат исторического события в двух источниках не считается новой попыткой. Число строк базы не является числом независимых экспериментов.

Для старых failed episodes без явного диагноза причина указана UNKNOWN. Аудит не переоценивал каждый контакт, не восстанавливал отсутствующий simulator state и не приписывал всем отказам последний диагноз. База не может включать недоступные или ранее удалённые артефакты. Историческое удаление отдельных диагностических записей по прежнему запросу пользователя отмечено в02; во время этого аудита данные не удалялись.

| Случай | Наблюдение | Причина и уверенность | Исправление и повторная проверка | Доказательство |
| --- | --- | --- | --- | --- |
'''
    for r in rows[:len(KNOWN)]:
        text += f"| {r['id']} | {r['symptom']} | {r['cause']} ({r['status']}) | {r['resolution']} | `{r['evidence']}` |\n"
    text += f'\nВ базе {len(rows)} строк, из них {sum(x.get("success") is False for x in raw)} отдельных raw manifest с отрицательной декларацией. Частотный анализ всех симптомов не заменяет причинный разбор: одни и те же сообщения guard могут возникать по разным причинам.\n'
    (OUT / '08_FAILURE_DATABASE.md').write_text(text)


def snapshot():
    old = read(OUT / 'system_snapshot.json')
    live = {}
    names = ['neural-recovery/924d1ee5ab01/state.json', 'room-learning/current.json',
             'room-learning/0c946789dbd7/state.json', 'runs/d4f59a63085b/job.json',
             'automatic-requests/queue-pause.json', 'diagnostics/early-room-current.json']
    for name in names:
        p = ROOT / name
        live[name] = dict(value=read(p), sha256=sha(p), modified=p.stat().st_mtime) if p.exists() else dict(missing=True)
    import psutil
    selected = []
    for process in psutil.process_iter(['pid', 'ppid', 'cmdline', 'cwd']):
        args = process.info['cmdline'] or []
        module = args[args.index('-m') + 1] if '-m' in args and args.index('-m') + 1 < len(args) else ''
        if module.startswith('simulator.'):
            selected.append(process.info)
    checkpoint = ROOT / 'runs/d4f59a63085b/model/checkpoints/029000'
    training = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))
                for p in checkpoint.rglob('*') if p.is_file()]
    result = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        audit_start_at=old['at'], runtime_revision=REV,
        documentation_base_revision=subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO,text=True).strip(),
        experiment_root=str(ROOT), disk_mounted=Path('/Volumes/ExplorerSim/env/bin/python').exists(),
        states=live, relevant_processes=selected, checkpoint029000_files=training,
        audit_actions=dict(training_updates=0, physics_rollouts=0, commands_to_robot=0,
                          weights_modified=False, datasets_modified=False, queues_resumed=False),
        pause_authorization='Explicit user request before forensic audit; not an audit-induced stop',
        caveat='Stale PID and nested cached state fields must be checked against real processes')
    dump(OUT / 'system_snapshot.json', result)
    (OUT / '17_CURRENT_SYSTEM_SNAPSHOT.md').write_text(f'''# 17. Состояние на окончание сбора доказательств

Срез UTC: **{result['at']}**. Полные состояния, хеши и команды оставшихся процессов — `system_snapshot.json`. Живые процессы проверены отдельно от JSON.

| Объект | Фактическое состояние |
| --- | --- |
| Диск | Подключён; обучение не возобновлялось |
| Neural recovery924d1ee5ab01 | paused, active_pids=[] |
| Room0c946789dbd7 | operator_paused, active_pids=[]; очередь одна |
| Learner d4f59a63085b | cancelled после пользовательской паузы; выполнено29078/145799 updates |
| Последнее полное сохранение | checkpoint029000; веса, optimizer, RNG, groups, training_step и processors сохранены |
| Несохранённый хвост | 78 updates после29000 не вошли в полный checkpoint; не объявлять их доступными для точного resume |
| Предыдущий learner1bd683a1d22b | 133577/133577; selected34de45050a00, local5/10 против2/10 reference |
| Последняя рассмотренная full-room диагностика | live-room-395c6a0365d9, те же scene1703000014, snapshot9000; отказjoint6 после695 решений, collision_free=true, grasp/lift/place=false |
| Полная квалификация | НЕ пройдена; нет100/100 и нет принятого активного ROSMASTER пакета |
| Реальный робот | Приводы не активировались; текущий read-only inventory недоступен по DNS; joint state command-only |

Неактуальные PID в marker/observer и nested cached `details.state=training` не являются доказательством работающего learner. Решающее сочетание: persistent pause, отсутствие процессов владельцев, state paused/cancelled, неизменённые полные checkpoints. В оставшихся процессах могут быть веб-службы, оболочки аудита и системные процессы; они не считаются обучением.

Аудит не останавливал полезное обучение: оно было отдельно приостановлено по прямой команде пользователя до начала исследования. После этой паузы аудит не запускал обучение, физические rollout или аппаратные команды и не менял веса/датасеты. Read-only CPU replay записывал только новые материалы аудита. Следующий запуск требует отдельного явного решения после разбора причин; этот пакет ничего автоматически не возобновляет.
''')


def index():
    entries = [
        ('E01', 'VERIFIED', 'История доступна с27.09; более ранняя часть не восстановлена', 'evidence/conversation_index.csv; evidence/git_history.tsv', '19;02'),
        ('E02', 'VERIFIED', 'Раннее ready относилось к компонентам, полного mobile executor не было', '19 D04 + source/docs/LEARNING-IMPLEMENTATION.ru.md', '19;02;09'),
        ('E03', 'VERIFIED', '15–17.10 parameter search временно не обучал ACT', 'source/simulator/skill_training.py;19 D15/D18', '04;19'),
        ('E04', 'VERIFIED', 'Реальные gradient runs и числовые checkpoints существуют', 'experiments.csv;checkpoints.csv;evidence/historical/checkpoint-007000-proof.json', '05;14;17'),
        ('E05', 'VERIFIED', 'Текущий набор739 records/629959 raw rows', 'dataset_statistics.json;evidence/recording_statistics.json;scripts/analyze_dataset.py', '06'),
        ('E06', 'VERIFIED', 'Command estimate nextstate равен прошлой action; это не measured joints', 'dataset_statistics.json;source/src/mobile_policy_contract.py; пользовательское уточнение', '03;06;07'),
        ('E07', 'VERIFIED', 'Плечо123° вне полного диапазона учебных наблюдений', 'evidence/runtime_distribution_followup.json;examples/refusal-695/actions.jsonl', '05;06'),
        ('E08', 'VERIFIED', 'Clock40.75s:728retreat/6unhook/1place; runtime остаётсяpick', 'evidence/runtime_distribution_followup.json', '05;06;07'),
        ('E09', 'VERIFIED', 'Fresh gripper target75.8205 и отказ при48° точно воспроизводятся', 'examples/refusal-695/refusal_forensics.json;source/simulator/room_policy.py::issued_command', '05'),
        ('E10', 'UNKNOWN', 'Единственная причинная доля clock/state/vision/teacher для скачка', 'examples/refusal-695/refusal_forensics.json counterfactuals and limitations', '05;11;12'),
        ('E11', 'VERIFIED', 'Четыре cross-split группы точных RGB в ограниченной выборке', 'evidence/visual_and_state_followup.json', '06'),
        ('E12', 'UNKNOWN', 'Полная semantic scene leakage и противоречивые targets всех кадров', 'visual sample scope2217 only; recording statistics', '06;15'),
        ('E13', 'VERIFIED', 'Teacher до3.6°/кадр против runtime2°; rawguard20°', 'dataset_statistics.json;source/src/mobile_policy_contract.py;source/simulator/room_policy.py', '07;05'),
        ('E14', 'VERIFIED', 'Timing repair10/10s не устранил последующий fresh refusal', 'evidence/historical/cadence-physics-regression.json;08 failures', '07;08'),
        ('E15', 'VERIFIED', 'Локальные5/10 не равны полной миссии', 'system_snapshot.json recovery history; selected full trial/result.json', '01;17'),
        ('E16', 'VERIFIED', 'Сохранены667 checkpointpaths/649 files/644 hashes', 'checkpoints.csv;evidence/checkpoint_inventory_summary.json', '14'),
        ('E17', 'VERIFIED', 'Пользовательская пауза действует;29000 полный,29078 job', 'system_snapshot.json', '17'),
        ('E18', 'NOT_IMPLEMENTED', 'Универсальное моторное обучение произвольной задаче из языка', 'source/simulator/language_learning.py;source/simulator/room_learning.py;09', '09;04'),
        ('E19', 'UNKNOWN', 'Реальные torque/lag/material/current latency и надёжность transfer', '03 таблица;15', '03;15'),
        ('E20', 'STRONGLY_SUPPORTED', 'Поздний контракт/сквозная проверка/потеря neuralconsumer объясняют существенную потерю времени', '26 decision chains19, run histories02, failure database08', '11;19'),
    ]
    dump(OUT / 'evidence/claim_index.json', [dict(zip(['id','status','claim','evidence','reports'],r)) for r in entries])
    text = '''# 18. Индекс доказательств

Каждая строка — ограниченный проверяемый вывод. VERIFIED не переносится с наличия кода на успех задачи. Для исходников точные функции и диапазоны строк с Git-ссылками находятся в `evidence/SOURCE_FUNCTION_INDEX.md`; содержимое и SHA — в `evidence/source_manifest.json`. Для run/checkpoint/scene используются собственные IDs первичных JSON, а не пересказ сообщения.

| ID | Статус | Вывод | Первичное доказательство | Разделы |
| --- | --- | --- | --- | --- |
'''
    for r in entries:
        text += '| ' + ' | '.join(r) + ' |\n'
    text += '\nХеш каждого включённого файла приведён в audit_manifest.json/SHA256SUMS. Исходные абсолютные пути вне архива — указатели на доступные владельцу данные; переносимый небольшой пример отказа уже включён. Прежние сообщения могут содержать ошибочные выводы: наличие цитаты доказывает сказанное, но не истинность технического утверждения.\n'
    (OUT / '18_EVIDENCE_INDEX.md').write_text(text)


def main():
    copy_evidence()
    failures()
    snapshot()
    index()
    print('Evidence, failures, snapshot and claim index assembled; no training actions')


if __name__ == '__main__':
    main()
