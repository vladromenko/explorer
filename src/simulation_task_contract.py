"""Read-only qualification of a portable simulation policy; no robot transport."""
import json
import hashlib
from pathlib import Path


def simulation_qualification(folder, bundle):
    path = Path(folder) / 'physics-validation.json'
    reasons = []
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError):
        report = {}
    if not isinstance(report,dict):
        report = {}
    checkpoints = bundle.get('validation_checkpoint_sha256', bundle['checkpoint_sha256'])
    manifest_path=Path(folder)/'source-transfer.json'
    if not manifest_path.is_file():
        manifest_path=Path(folder)/'transfer.json'
    if manifest_path.is_file():
        manifest=json.loads(manifest_path.read_text())
        expected=manifest.get('files',{}).get('physics-validation.json')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            reasons.append('Отчёт проверки изменён после упаковки')
    if report.get('validation_kind') != 'closed_loop_policy':
        reasons.append('Нет проверки обученной политики в замкнутом цикле')
    if report.get('model_weights_sha256') != bundle['checkpoint_sha256'].get('model.safetensors') or report.get('model_checkpoint_sha256') != checkpoints:
        reasons.append('Проверка относится к другой модели или настройкам исполнения')
    if report.get('heldout_independent') is not True:
        reasons.append('Нет независимых сцен проверки')
    if report.get('physics_validated') is not False or report.get('real_hardware_accepted') is not False:
        reasons.append('Симуляция не подтверждает приёмку динамики или настоящего робота')
    episodes, successes = report.get('episodes',0), report.get('successes',0)
    if type(episodes) is not int or type(successes) is not int or episodes < 10 or successes != episodes:
        reasons.append('Нужны минимум 10 успешных независимых проверок подряд')
    results = report.get('results',[])
    results_valid = isinstance(results,list) and len(results)==episodes and all(isinstance(item,dict) and type(item.get('success')) is bool for item in results)
    if not results_valid or sum(item['success'] is True for item in results)!=successes:
        reasons.append('Итоговый счётчик не подтверждён отдельными результатами')
    if bundle.get('skill_id') and report.get('task_id') != bundle['skill_id']:
        reasons.append('Проверка относится к другой задаче')
    if bundle.get('scene_profile') == 'sock_variants_v1':
        from simulator.domain_randomization import scene_coverage
        coverage = scene_coverage(results) if results_valid else {}
        if (not coverage.get('all_height_categories_present') or coverage.get('episodes') != episodes
            or coverage.get('distinct_seeds',0) < episodes
            or coverage.get('user_basket_25cm_diameter_30cm_rim_episodes',0) < 1):
            reasons.append('Независимая проверка не покрывает все высоты корзины нового профиля')
    return dict(simulation_qualified=not reasons, blocked_by=reasons,
                successes=successes, episodes=episodes, physical_accepted=False,
                observed_transfer_validation_required=True,
                autonomous_execution_allowed=False)
