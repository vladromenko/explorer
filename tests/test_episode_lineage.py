import pytest
from simulator.episode_lineage import recording_index,trained_recording
from simulator.learning import sha256


def test_preserved_interrupted_copy_cannot_hide_the_exact_training_recording(tmp_path):
    identity='recovery-example';base=tmp_path/'neural-recovery/round-owner/round-01'
    correct=base/'correction-09'/identity;interrupted=base/'correction-09-interrupted-123'/identity
    for p,value in ((correct,b'exact training input'),(interrupted,b'different preserved input')):
        p.mkdir(parents=True);(p/'samples.jsonl').write_bytes(value);(p/'episode.json').write_text('{}')
    row={'episode':{'id':identity},'input_sha256':{'samples.jsonl':sha256(correct/'samples.jsonl')}}
    index=recording_index(tmp_path)
    assert len(index[identity])==2
    assert trained_recording(row,index)==correct
    assert trained_recording(row,{identity:list(reversed(index[identity]))})==correct
    row['input_sha256']['samples.jsonl']=sha256(interrupted/'samples.jsonl')
    assert trained_recording(row,index)==interrupted
    row['input_sha256']['samples.jsonl']='0'*64
    with pytest.raises(ValueError,match='No original recording matches'):trained_recording(row,index)
    assert correct.exists() and interrupted.exists()
