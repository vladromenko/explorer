"""Cause-directed executor repairs, qualified by paired real room trials.

Only checked behavioural operators can change. Physics, perception confidence,
motion protection and outcome criteria are never repairable parameters.
"""
OPERATORS=('expanded_coverage','expanded_approaches')


def program(value):
    if not isinstance(value,list) or any(type(v) is not str for v in value) or len(value)!=len(set(value)) or any(v not in OPERATORS for v in value):
        raise ValueError('Unknown room executor recovery program')
    return sorted(value)


def propose(current,diagnosis):
    selected=set(program(current));before=set(selected)
    for row in diagnosis['findings']:
        reason=row['reason'].lower()
        # These identify exhausted *checked proposals*, not permission to
        # move after a safety guard, camera epoch change or STOP.
        if row['cause']=='search_or_perception' and any(s in reason for s in (
                'search budget exhausted','no unvisited reachable observation point')):
            selected.add('expanded_coverage')
        if row['cause']=='alignment_or_reacquisition' and any(s in reason for s in (
                'fresh geometric alignment did not converge','no collision-checked approach',
                'target not confirmed after navigation')):
            selected.add('expanded_approaches')
    return sorted(selected) if selected!=before else None


def progress(row):
    """Observed completed milestones; never advertised as physical success."""
    done=set()
    for event in row.get('events',[]):
        stage=event['stage']
        if stage=='search' and set((event.get('result') or {}).get('found',[]))=={'sock','basket'}:done.add('search')
        elif stage in ('approach_sock','policy_pick','verify_hold','approach_basket','policy_place','verify_place'):done.add(stage)
    stages=('search','approach_sock','policy_pick','verify_hold','approach_basket','policy_place','verify_place')
    count=0
    for stage in stages:
        if stage not in done:break
        count+=1
    return count


def comparison(before,after):
    left={r['seed']:r for r in before['results']};right={r['seed']:r for r in after['results']}
    if len(before['results'])!=10 or len(after['results'])!=10 or len(left)!=10 or set(left)!=set(right):
        raise ValueError('Executor repair needs ten identical paired scenes')
    regressions=[];improvements=[]
    for seed,old in left.items():
        new=right[seed]
        if (new.get('collision_free') is not True or new.get('cancelled') or new.get('stop_error')
                or old.get('success') is True and new.get('success') is not True
                or progress(new)<progress(old)):
            regressions.append(seed)
        if (new.get('success') is True and old.get('success') is not True) or progress(new)>progress(old):improvements.append(seed)
    return dict(accepted=bool(improvements) and not regressions,improved_scene_ids=improvements,
        regressions=regressions,comparison_kind='same_model_same_scenes_executor_only',
        neural_training=False,simulation_qualified=False,physical_ready=False)
