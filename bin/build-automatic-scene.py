#!/usr/bin/env python3
"""Create an actual immutable physics scene and preview from a checked recipe."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
import mujoco
from simulator.environment import ExplorerEnv
from simulator.scene_recipes import validate_recipe,reset_options
from simulator.learning import write_json

folder=Path(sys.argv[1]);recipe=validate_recipe(json.loads((folder/'recipe.json').read_text()))
env=ExplorerEnv()
try:
    env.reset(seed=906160004,options=reset_options(recipe,906160004))
    Image.fromarray(env.render_rgb('overview')).save(folder/'preview.jpg')
    mujoco.mj_saveLastXML(str(folder/'scene.xml'),env.model)
    write_json(folder/'scene.json',dict(schema=1,recipe_sha256=recipe['sha256'],task_id=recipe['task_id'],
        files={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ('scene.xml','preview.jpg','recipe.json')},
        estimated_dynamics=True,actor_uses_scene_coordinates=False,physical_ready=False))
finally:env.close()
