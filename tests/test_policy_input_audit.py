import numpy as np

from simulator.checkpoint_experiment import shortlist,training_seeds
from simulator.policy_input_audit import interventions


def test_shortlist_covers_training_exposure_instead_of_only_adjacent_low_mae_steps():
    selection={'candidates':[{'checkpoint_relative':f'model/checkpoints/{i:06d}/pretrained_model',
        'normalized_mae':1/i} for i in range(1000,9000,1000)]}
    selected=shortlist(selection)
    assert [int(p.split('/')[2]) for p in selected]==[2000,4000,6000,8000]


def test_one_input_is_intervened_at_a_time_and_originals_are_preserved():
    class Tensor(np.ndarray):
        def roll(self,shift,dims):return np.roll(self,shift,axis=dims).view(Tensor)
    inputs={name:np.arange(8).reshape(4,2).view(Tensor) for name in
        ('observation.state','observation.images.wrist','clock')}
    before={key:value.copy() for key,value in inputs.items()}
    variations=interventions(inputs,'clock')
    for name,key in [('shuffled_image','observation.images.wrist'),('shuffled_state','observation.state'),('shuffled_clock','clock')]:
        assert not np.array_equal(variations[name][key],before[key])
        for other in inputs:
            if other!=key:assert np.array_equal(variations[name][other],before[other])
    for key in inputs:assert np.array_equal(inputs[key],before[key])
    assert not variations['black_image']['observation.images.wrist'].any()


def test_selection_scene_lineage_includes_recorded_validation(tmp_path):
    import json
    for kind,seed in [('train',1),('validation',2)]:
        folder=tmp_path/kind;folder.mkdir()
        (folder/'explorer-provenance.json').write_text(json.dumps({'episodes':[{'episode':{'seed':seed}}]}))
    assert training_seeds(tmp_path)=={1,2}
