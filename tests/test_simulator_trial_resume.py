import json
from pathlib import Path

from simulator.learning import write_json
from simulator.trial_resume import completed_attempts
from src.mobile_policy_contract import CONTRACT_SHA256


def recording(root, name, seed, success=False, error=None, marker=True, finished=100):
    folder = root / 'demonstrations' / name
    folder.mkdir(parents=True)
    episode = dict(id=name, source='simulation', generator='ground_truth_teacher',
        task_id='sock_basket', scene_profile='sock_variants_v1', seed=seed,
        model_provenance={'model': 'accepted'}, finished_at=finished, frames=3,
        success=success, error=error, fps=20, sample_contract_sha256=CONTRACT_SHA256)
    if marker is not None:
        episode['attempt_completed'] = marker
    write_json(folder / 'episode.json', episode)
    rows = []
    for index in range(3):
        (folder / f'{index}.jpg').write_bytes(b'frame')
        rows.append(dict(at=100+index*.05, observation=[90]*6+[0]*3,
                         action=[90]*6+[0]*3, wrist_rgb=f'{index}.jpg'))
    (folder / 'samples.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    return folder


def recover(root):
    return completed_attempts(root, 'sock_basket', 'sock_variants_v1', 100, 5,
                              {'model': 'accepted'})


def test_resume_keeps_completed_failures_and_retries_interrupted_legacy_attempt(tmp_path):
    recording(tmp_path, 'trial-a', 100, success=True, marker=None)
    recording(tmp_path, 'trial-b', 101, error='Контактное удержание не подтверждено', marker=None)
    recording(tmp_path, 'trial-c', 102, error='Задание остановлено', marker=None)
    recording(tmp_path, 'trial-d', 103, marker=False)
    found = recover(tmp_path)
    assert set(found) == {0, 1}
    assert found[1]['success'] is False


def test_resume_rejects_wrong_world_and_corrupt_recording(tmp_path):
    folder = recording(tmp_path, 'trial-a', 100)
    (folder / '1.jpg').unlink()
    wrong = recording(tmp_path, 'trial-b', 101)
    path = wrong / 'episode.json'
    episode = json.loads(path.read_text())
    episode['model_provenance'] = {'model': 'changed'}
    write_json(path, episode)
    recording(tmp_path, 'trial-c', 999)
    assert recover(tmp_path) == {}


def test_repeated_scene_is_not_selected_by_success(tmp_path):
    recording(tmp_path, 'trial-first', 100, success=False, finished=100)
    recording(tmp_path, 'trial-repeat', 100, success=True, finished=200)
    assert recover(tmp_path)[0]['id'] == 'trial-first'


def test_resume_rejects_wrong_task_profile_and_incomplete_sample_count(tmp_path):
    for name, changed in (('trial-a', {'task_id': 'toy_box'}),
                          ('trial-b', {'scene_profile': 'baseline'}),
                          ('trial-c', {'frames': 4})):
        folder = recording(tmp_path, name, 100)
        path = folder / 'episode.json'
        write_json(path, dict(json.loads(path.read_text()), **changed))
    assert recover(tmp_path) == {}
