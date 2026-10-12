"""Recover completed teacher attempts without treating interrupted ones as done."""
import json
from pathlib import Path

from .learning import inspect_episode


def completed_attempts(root, task_id, profile, first_seed, count, provenance):
    profile = profile if task_id in ('sock_basket', 'cloth_basket') else 'baseline'
    candidates = []
    for path in (Path(root) / 'demonstrations').glob('trial-*/episode.json'):
        try:
            episode = json.loads(path.read_text())
            seed = episode.get('seed')
            legacy_complete = ('attempt_completed' not in episode and
                               episode.get('error') not in ('Задание остановлено', 'Simulator training cancelled'))
            matches = (type(seed) is int and first_seed <= seed < first_seed + count
                       and episode.get('id') == path.parent.name
                       and episode.get('task_id', 'sock_basket') == task_id
                       and episode.get('scene_profile', 'baseline') == profile
                       and episode.get('source') == 'simulation'
                       and episode.get('generator') == 'ground_truth_teacher'
                       and episode.get('model_provenance') == provenance
                       and type(episode.get('finished_at')) in (int, float)
                       and type(episode.get('success')) is bool
                       and (episode.get('attempt_completed') is True or legacy_complete))
            if matches:
                candidates.append((episode['finished_at'], path.parent, episode))
        except (OSError, ValueError, AttributeError, TypeError):
            pass
    recovered = {}
    # Keep the earliest complete attempt, including failures. Selecting only
    # successful repeats would distort the reported success rate.
    for _, folder, episode in sorted(candidates, key=lambda item: (item[0], item[1].name)):
        index = episode['seed'] - first_seed
        if index not in recovered:
            try:
                _, rows, _, _ = inspect_episode(folder)
                if type(episode.get('frames')) is int and len(rows) == episode['frames']:
                    recovered[index] = episode
            except (OSError, ValueError, KeyError, TypeError):
                pass
    return recovered
