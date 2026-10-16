"""Declarative scene construction; language never generates executable code.

These recipes cover a declared object family and measured robot model. Unknown
objects/actions are refused rather than represented as a different object.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Literal

from pydantic import BaseModel,ConfigDict,Field,model_validator

from .curriculum import TASKS,task_spec
from .domain_randomization import sample_sock_scene


class SceneRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    task_id:Literal['sock_basket','cloth_basket','toy_box','cookie_delivery']
    basket_height_cm:float|None=Field(default=None,ge=8,le=30)
    basket_diameter_cm:float|None=Field(default=None,ge=20,le=40)
    basket_width_cm:float|None=Field(default=None,ge=20,le=45)
    basket_depth_cm:float|None=Field(default=None,ge=17,le=40)
    object_length_cm:float|None=Field(default=None,ge=2,le=20)
    object_width_cm:float|None=Field(default=None,ge=2,le=8)

    @model_validator(mode='after')
    def validate_family(self):
        if self.task_id not in ('sock_basket','cloth_basket') and any(v is not None for v in
                (self.basket_height_cm,self.basket_diameter_cm,self.basket_width_cm,self.basket_depth_cm)):
            raise ValueError('Basket dimensions cannot be applied to a tray or toy box')
        if self.basket_diameter_cm is not None and (self.basket_width_cm is not None or self.basket_depth_cm is not None):
            raise ValueError('Укажите диаметр круглой корзины либо ширину и глубину прямоугольной')
        if self.basket_height_cm is not None and self.basket_height_cm not in (8,12,16,20,30):
            raise ValueError('Basket heights currently supported: 8,12,16,20,30 cm')
        if self.task_id in ('sock_basket','cloth_basket'):
            if self.object_length_cm is not None and not 12<=self.object_length_cm<=20:
                raise ValueError('Cloth length currently supported: 12–20 cm')
            if self.object_width_cm is not None and not 4<=self.object_width_cm<=6.5:
                raise ValueError('Cloth width currently supported: 4–6.5 cm')
        elif self.object_length_cm is not None or self.object_width_cm is not None:
            raise ValueError('Custom rigid-object dimensions require a matching physical asset')
        return self


def compile_recipes(plan,scene_requests=()):
    if not plan.get('fully_supported'):
        return dict(state='needs_clarification',recipes=[],missing=plan.get('missing',[]),
            question=plan.get('clarification') or 'Уточните неподдерживаемые части задачи.')
    requests=[SceneRequest.model_validate(row) for row in scene_requests]
    by_task={r.task_id:r for r in requests}
    if len(by_task)!=len(requests) or set(by_task)-set(plan['task_ids']):
        raise ValueError('Scene requirements must correspond uniquely to requested tasks')
    recipes=[]
    for task in plan['task_ids']:
        spec=task_spec(task);request=by_task.get(task,SceneRequest(task_id=task))
        recipe=dict(schema=1,task_id=task,title=spec['title'],query=TASKS[task]['query'],
            scene=spec['scene'],layout='two_rooms_doorway' if spec['rooms'] else 'single_room',
            request=request.model_dump(exclude_none=True),estimated_dynamics=True,
            semantic_recognition_required=True,hardware_ready=False,
            scope='one visible object per episode; real room map and handover are separate',
            steps=spec['recipe'])
        recipe['sha256']=recipe_digest(recipe);recipes.append(recipe)
    return dict(state='compiled',recipes=recipes,missing=[],question='')


def recipe_digest(recipe):
    body={k:v for k,v in recipe.items() if k!='sha256'}
    return hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def validate_recipe(recipe):
    if not isinstance(recipe,dict) or recipe.get('schema')!=1 or recipe.get('sha256')!=recipe_digest(recipe):
        raise ValueError('Scene recipe checksum mismatch')
    request=SceneRequest.model_validate(recipe['request'])
    spec=task_spec(request.task_id)
    if (recipe.get('task_id')!=request.task_id or recipe.get('scene')!=spec['scene'] or
        recipe.get('query')!=spec['query'] or recipe.get('semantic_recognition_required') is not True or
        recipe.get('layout')!=('two_rooms_doorway' if spec['rooms'] else 'single_room') or
        recipe.get('steps')!=spec['recipe'] or recipe.get('hardware_ready') is not False):
        raise ValueError('Recipe does not match its executable object family')
    return copy.deepcopy(recipe)


def reset_options(recipe,seed,*,validation=False):
    recipe=validate_recipe(recipe);task=recipe['task_id'];request=SceneRequest.model_validate(recipe['request'])
    options=dict(task_id=task,difficulty='varied',scene_profile='sock_variants_v1' if task in ('sock_basket','cloth_basket') else 'baseline')
    if task in ('sock_basket','cloth_basket'):
        scene=sample_sock_scene(seed)
        # Standard independent validation retains the entire original height
        # coverage. Extra user-specific cases are evaluated separately.
        if not validation:
            if request.basket_height_cm is not None:scene['parameters']['bin']['height_m']=request.basket_height_cm/100
            if request.basket_diameter_cm is not None:
                radius=request.basket_diameter_cm/200
                scene['parameters']['bin'].update(shape='round',inner_half_size_m=[radius,radius])
            if request.basket_width_cm is not None or request.basket_depth_cm is not None:
                scene['parameters']['bin']['shape']='rectangular'
                size=scene['parameters']['bin']['inner_half_size_m']
                if request.basket_width_cm is not None:size[0]=request.basket_width_cm/200
                if request.basket_depth_cm is not None:size[1]=request.basket_depth_cm/200
            if request.object_length_cm is not None:scene['parameters']['sock']['length_m']=request.object_length_cm/100
            if request.object_width_cm is not None:scene['parameters']['sock']['width_m']=request.object_width_cm/100
        # Scene generation is separate from actor observations. These values
        # are accepted only by reset, never by the manipulation port.
        scene['bin_position'][1]=scene['parameters']['bin']['inner_half_size_m'][1]+.20
        options['generated_scene']=scene
    return options


def explicit_scene_requests(text,task_ids,proposals=()):
    """Numbers/units come from the user's text, never invented LLM defaults."""
    import re
    values={task:{'task_id':task} for task in task_ids}
    unit=r'(?:сантиметр\w*|миллиметр\w*|метр\w*|см|мм|м)(?![а-яa-z])'
    pattern=re.compile(r'(?P<label>высот\w*|бортик\w*|диаметр\w*|ширин\w*|глубин\w*|длин\w*)\s*(?:(?:корзин\w*|носк\w*|ткан\w*)\s*)?(?:около\s*|примерно\s*|[:=]\s*)?(?P<number>\d+(?:[.,]\d+)?)\s*(?P<unit>'+unit+r')',re.I)
    for turn in text.split('\nУточнение пользователя:'):
        for match in pattern.finditer(turn):
            value=float(match['number'].replace(',','.'));u=match['unit'].lower()
            if u.startswith('мм') or u.startswith('милли'):value/=10
            elif u=='м' or u.startswith('метр'):value*=100
            prefix=turn[:match.start()].lower();mentions=list(re.finditer(r'корзин\w*|носк\w*|носок|ткан\w*|тряп\w*|поднос\w*|короб\w*',prefix))
            noun=mentions[-1][0] if mentions else 'корзина'
            label=match['label'].lower()
            basket=noun.startswith('корзин')
            if basket:
                keys={'высот':'basket_height_cm','бортик':'basket_height_cm','диаметр':'basket_diameter_cm','ширин':'basket_width_cm','глубин':'basket_depth_cm'}
                key=next((field for stem,field in keys.items() if label.startswith(stem)),None)
                applicable=[task for task in task_ids if task in ('sock_basket','cloth_basket')]
            elif noun.startswith(('носк','носок','ткан','тряп')):
                key='object_length_cm' if label.startswith('длин') else 'object_width_cm' if label.startswith('ширин') else None
                applicable=[task for task in task_ids if task in ('sock_basket','cloth_basket')]
            else:raise ValueError('Указанные размеры требуют отдельной физической модели подноса или коробки')
            if key is None:raise ValueError('Уточните, к какому предмету относится размер: '+match[0])
            for task in applicable:values[task][key]=value
    # Proposals may help identify a task, but may not add measurements absent
    # from the explicit number/unit parser above.
    return [SceneRequest.model_validate(value).model_dump(exclude_none=True) for value in values.values() if len(value)>1]
