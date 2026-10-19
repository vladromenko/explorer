"""Build and verify a bounded evidence archive; never packages large weights."""
import datetime
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'ROSMASTER_INDEPENDENT_AUDIT.zip'
EXCLUDE = {'audit_manifest.json', 'SHA256SUMS', ARCHIVE.name, ARCHIVE.name+'.sha256', '.DS_Store'}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    paths = [p for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name not in EXCLUDE
             and '__pycache__' not in p.parts and p.suffix != '.pyc']
    if any(p.name in ['model.safetensors','optimizer_state.safetensors','rng_state.safetensors'] for p in paths):
        raise ValueError('Heavy model/training state must remain outside packet')
    files = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=digest(p)) for p in paths]
    manifest = dict(schema='rosmaster_independent_audit_v1',
        built_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        runtime_revision='1da8ecb26efaa798d4293f126d7d123aff7985e5',
        documentation_parent_revision='bded49e11fa075c75189da7f505d5f12552cfbfa',
        audit_commit='Exact committed documentation revision is supplied in delivery; manifest cannot hash its own Git commit',
        scope=dict(conversation_messages=2227, first_available_day_msk='2026-09-27',
            current_recordings_full_streams=739, current_raw_rows=629959, sampled_images=2217,
            all_historical_images_decoded=False, checkpoint_weight_files_hashed=649,
            technical_statuses=['VERIFIED','STRONGLY_SUPPORTED','HYPOTHESIS','UNKNOWN','NOT_IMPLEMENTED']),
        safety=dict(training_resumed=False, weights_modified=False,datasets_deleted=False,
                    robot_actuation=False, full_conversation_excluded_from_git=True),
        omissions=['Heavy model weights, datasets and optimizer tensors; see checkpoints.csv',
                   'Unavailable history before27Sep and unavailable full simulator state at refusal695',
                   'Live robot inventory/latency and unmeasured physical parameters',
                   'Hidden reasoning and complete historical tool stdout'],
        files=files)
    (ROOT/'audit_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    sums=files+[dict(path='audit_manifest.json',sha256=digest(ROOT/'audit_manifest.json'))]
    (ROOT/'SHA256SUMS').write_text(''.join(row['sha256']+'  '+row['path']+'\n' for row in sums))
    payload=paths+[ROOT/'audit_manifest.json',ROOT/'SHA256SUMS']
    with zipfile.ZipFile(ARCHIVE,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in payload:
            z.write(p, str(p.relative_to(ROOT)))
    with zipfile.ZipFile(ARCHIVE) as z:
        bad=z.testzip()
        if bad:
            raise ValueError('Corrupt ZIP member: '+bad)
        for row in files:
            if hashlib.sha256(z.read(row['path'])).hexdigest()!=row['sha256']:
                raise ValueError('Archived hash mismatch: '+row['path'])
    archive_sha=digest(ARCHIVE)
    (ROOT/(ARCHIVE.name+'.sha256')).write_text(archive_sha+'  '+ARCHIVE.name+'\n')
    print(json.dumps(dict(files=len(payload),payload_bytes=sum(p.stat().st_size for p in payload),
        zip_bytes=ARCHIVE.stat().st_size,zip_sha256=archive_sha,verified=True)))


if __name__=='__main__':
    main()
