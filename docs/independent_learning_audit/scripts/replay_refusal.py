"""Reproduce inference on saved RGB only; no environment, training or actuators."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from simulator.learning import load_policy, clock_features, sha256
    from simulator.room_policy import issued_command
    from simulator.environment import LOW, HIGH
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    trace = next((args.diagnostic / 'trial/observed').glob('policy-pick-*/actions.jsonl'))
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    refusal = next(r for r in reversed(rows) if r.get('status') == 'refused')
    step = refusal['step']
    observation = next(r for r in rows if r.get('event') == 'observation' and r['step'] == step)
    image_path = trace.parent / observation['image']
    rgb = np.asarray(Image.open(image_path).convert('RGB'))
    command = np.asarray(observation['command'], dtype=np.float32)
    clock = refusal['decision']['elapsed_s']
    if hashlib.sha256(rgb.tobytes()).hexdigest() != observation['rgb_sha256']:
        raise ValueError('Recorded image bytes do not match exposure')
    torch.set_num_threads(1)
    torch.manual_seed(0)
    actor = load_policy(args.diagnostic / 'model', device='cpu')
    context = actor.bundle['actor_context']

    def forward(elapsed, image):
        actor.reset()
        raw = {'observation.state': torch.tensor(command[None]),
               'observation.images.wrist': torch.from_numpy(image.copy()).permute(2, 0, 1).float()[None] / 255,
               context['feature']: torch.tensor([clock_features(context, elapsed)], dtype=torch.float32)}
        batch = actor.pre(raw)
        with torch.no_grad():
            normalized_chunk = actor.policy.predict_action_chunk(batch)
            endpoints = torch.stack([actor.post(normalized_chunk[:, i].clone())
                                     for i in range(normalized_chunk.shape[1])], dim=1)
        return raw, batch, normalized_chunk.cpu().numpy(), endpoints.cpu().numpy()

    raw, processed, normalized, endpoints = forward(clock, rgb)
    np.savez_compressed(out / 'normalized_inputs.npz', **{
        k.replace('.', '_'): v.detach().cpu().numpy() for k, v in processed.items()
        if isinstance(v, torch.Tensor)})
    predicted = endpoints[0, 0]
    expected = np.asarray(refusal['decision']['prediction'])
    error = float(np.max(abs(predicted - expected)))
    try:
        issued_command(command, np.clip(predicted, LOW, HIGH), np.zeros(6))
        guard = dict(accepted=True)
    except ValueError as exc:
        guard = dict(accepted=False, reason=str(exc))
    ablations = []
    for elapsed in [0., 6., 12., 24., 35., clock, 43.55, 50.]:
        _, _, _, action = forward(elapsed, rgb)
        ablations.append(dict(kind='clock', clock_s=elapsed, first_endpoint=action[0, 0].tolist(),
                              delta=(action[0, 0, :6] - command[:6]).tolist()))
    for label, image in [('black_rgb', np.zeros_like(rgb)), ('gray_rgb', np.full_like(rgb, 127))]:
        _, _, _, action = forward(clock, image)
        ablations.append(dict(kind=label, clock_s=clock, first_endpoint=action[0, 0].tolist(),
                              delta=(action[0, 0, :6] - command[:6]).tolist()))
    selected = []
    for row in rows:
        if row.get('event') == 'observation' and row['step'] in [0, 500, 650, 675, 690, 694, 695]:
            source = trace.parent / row['image']
            shutil.copy2(source, out / row['image'])
            selected.append(row)
    (out / 'actions.jsonl').write_text(trace.read_text())
    result = dict(schema='read_only_refusal_forensics_v1', source_trace=str(trace),
        trace_sha256=sha256(trace), checkpoint_sha256=actor.bundle['checkpoint_sha256'],
        recorded_refusal=refusal, observation=observation, clock_features=clock_features(context, clock),
        normalized_state=processed['observation.state'].cpu().numpy().tolist(),
        normalized_clock=processed[context['feature']].cpu().numpy().tolist(),
        image_summary=dict(shape=list(rgb.shape), minimum=int(rgb.min()), maximum=int(rgb.max())),
        full_normalized_chunk=normalized.tolist(), full_denormalized_chunk=endpoints.tolist(),
        max_abs_reproduction_error=error, replay_matches_atol_1e_4=error < 1e-4,
        guard=guard, counterfactuals=ablations, selected_exposures=selected,
        teacher_target_at_runtime_state=None, full_original_qpos_qvel_at_refusal=None,
        limitations=['No recorded expert counterfactual at this student state',
                    'Original full MuJoCo state was not saved at every action',
                    'Inference counterfactuals are not physical rollouts and cannot establish task success'],
        commands_sent=False, training_updates=0, source_files_modified=False)
    (out / 'refusal_forensics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(reproduction_error=error, replay_matches=result['replay_matches_atol_1e_4'],
                         recorded_gripper=float(expected[5]), replay_gripper=float(predicted[5]), guard=guard)))


if __name__ == '__main__':
    main()
