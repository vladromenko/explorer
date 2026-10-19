"""Recheck complete saved streams; output must be a new file outside experiments."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--packet', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(Path('/Volumes/ExplorerSim/ExplorerSim')):
        raise ValueError('Use a new output file outside the experiment tree')
    records = json.loads((args.packet/'evidence/recording_statistics.json').read_text())
    maximum = [float('-inf')] * 9
    minimum = [float('inf')] * 9
    phases = Counter()
    high = 0
    for record in records:
        path = Path(record['source'])/'samples.jsonl'
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record['samples_sha256']:
            raise ValueError('Recording bytes changed: ' + record['id'])
        rows = [json.loads(line) for line in payload.splitlines()]
        for i,row in enumerate(rows):
            if row.get('training_target_valid', True):
                state = row['observation']
                maximum = [max(x,y) for x,y in zip(maximum,state)]
                minimum = [min(x,y) for x,y in zip(minimum,state)]
                high += int(state[1] >= 123.)
                # Match export_dataset's actor clock, not absolute physics time.
                clock = row.get('actor_elapsed_s', i*.05)
                if abs(clock-40.75) < .026:
                    phases[row.get('phase','unknown')] += 1
    result = dict(recordings=len(records),valid_observation_min=minimum,valid_observation_max=maximum,
        valid_observations_shoulder_ge_123=high,training_phases_at_40_75_s=dict(phases),
        commands_sent=False,training_updates=0)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    expected = json.loads((args.packet/'evidence/runtime_distribution_followup.json').read_text())
    keys = ['valid_observation_min','valid_observation_max','valid_observations_shoulder_ge_123',
            'training_phases_at_40_75_s']
    if not all(result[k] == expected[k] for k in keys):
        raise ValueError('Recheck differs; inspect output without overwriting original evidence')
    print('All complete-stream range and phase counts reproduced')


if __name__ == '__main__':
    main()
