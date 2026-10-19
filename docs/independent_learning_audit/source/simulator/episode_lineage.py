"""Locate the exact trained recording, including preserved interrupted copies.

An episode ID is not unique across attempts retained after an interruption.
The training provenance's content digest, not glob order, chooses its input.
"""
from pathlib import Path
from .learning import sha256


def recording_index(root):
    index={}
    for pattern in ('demonstrations/*/episode.json',
            'neural-recovery/*/round-*/correction-*/recovery-*/episode.json',
            'room-learning/*/round-*/corrections/scene-*/room-expert-*/episode.json'):
        for path in Path(root).glob(pattern):index.setdefault(path.parent.name,[]).append(path.parent)
    return index


def trained_recording(row,index,cache=None):
    identity=row['episode']['id'];expected=row['input_sha256']['samples.jsonl']
    candidates=index.get(identity,[]);cache={} if cache is None else cache
    for path in sorted(candidates):
        path=Path(path);sample=path/'samples.jsonl'
        if sample not in cache:cache[sample]=sha256(sample)
        if cache[sample]==expected:return path
    raise ValueError('No original recording matches trained checksum: '+identity+
        '; preserved candidates: '+', '.join(str(p) for p in candidates))
