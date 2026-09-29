import json
from pathlib import Path

ROOT=Path(__file__).parents[1]

def test_phone_panel_is_separate_and_has_finite_arm_and_releasing_drive():
    page=(ROOT/'src/mobile.html').read_text()
    assert "arm/jog" in page and "manual_release" in page and "teaching/mobile/start" in page
    assert "visibilitychange" in page and "lastFrame" in page

def test_training_goals_are_unique_and_bounded():
    data=json.loads((ROOT/'config/mobile-training-goals.json').read_text())
    ids=[g['id'] for g in data['goals']]
    assert len(ids)==len(set(ids))>=5
    allowed={'soft_cloth','rigid','fragile','slippery','deformable','unknown'}
    assert all(g['object_class'] in allowed and g['size_class'] in {'small','medium','large'} for g in data['goals'])
