#!/usr/bin/env python3
"""Local acceptance: authenticated read-only experiments, STOP latency; no movement."""
import json
from pathlib import Path
import statistics
import threading
import time
import urllib.request
import urllib.error
import uuid
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

root=Path('/home/vlad/Explorer')
token=(root/'config/access_token').read_text().strip()
def api(path,payload=None,auth=True):
    request=urllib.request.Request('http://127.0.0.1:8080/api/'+path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Authorization':'Bearer '+(token if auth else 'invalid'),'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=20) as response:return response.status,json.load(response)
    except urllib.error.HTTPError as exc:return exc.code,json.load(exc)
def percentile(values,q):return sorted(values)[min(len(values)-1,int(q*(len(values)-1)))]
def stats(values):return dict(count=len(values),p50_ms=statistics.median(values),p95_ms=percentile(values,.95),max_ms=max(values))

assert api('experiments',auth=False)[0]==401
catalog=api('experiments')[1];assert len(catalog['experiments'])==18
results=[]
for entry in catalog['experiments']:
    payload=dict(experiment=entry['id'],mode='observe',params={},request_id=uuid.uuid4().hex,issued_at=time.time())
    code,r=api('experiments/run',payload)
    assert code==200,(entry['id'],r)
    results.append(dict(experiment=entry['id'],state=r['state'],reason=r.get('result',{}).get('summary'),id=r['id']))
    assert api('experiments/run',payload)[1]['id']==r['id']
physical=dict(experiment='E15',mode='physical',params={},request_id=uuid.uuid4().hex,issued_at=time.time())
assert api('experiments/run',physical)[0]==409
physical.update(mode='observe',issued_at=time.time()-40)
assert api('experiments/run',physical)[0]==409

rclpy.init();node=Node('explorer_lab_latency_check');acks={}
node.create_subscription(String,'/explorer/request_ack',lambda m:acks.update({json.loads(m.data)['id']:json.loads(m.data)}),10)
thread=threading.Thread(target=rclpy.spin,args=(node,),daemon=True);thread.start()
deadline=time.monotonic()+8
while node.count_publishers('/explorer/request_ack')==0 and time.monotonic()<deadline:time.sleep(.1)
assert node.count_publishers('/explorer/request_ack')>0,'STOP acknowledgement publisher unavailable'
# DDS discovery does not imply the writer has matched this subscriber yet.
warm_id=api('control',dict(op='stop'))[1]['id']
deadline=time.monotonic()+5
while warm_id not in acks and time.monotonic()<deadline:time.sleep(.01)
assert warm_id in acks,'STOP acknowledgement subscriber not matched'
api_times=[];stop_times=[]
for i in range(20):
    begin=time.monotonic();code,_=api('status');api_times.append((time.monotonic()-begin)*1000)
    assert code==200
    begin=time.monotonic();code,result=api('control',dict(op='stop'))
    assert code==200
    deadline=time.monotonic()+1
    while result['id'] not in acks and time.monotonic()<deadline:time.sleep(.002)
    assert result['id'] in acks,'No core STOP acknowledgement'
    stop_times.append((acks[result['id']]['handled_monotonic']-begin)*1000)
state=api('status')[1]
assert state['stop_latched'] and not any(state['velocity'])
report=dict(at=time.time(),experiments=results,local_status=stats(api_times),software_stop=stats(stop_times),
            physical_braking_measured=False,resources=state.get('resources'),sensor_age=state.get('sensor_age'),
            battery_voltage=state.get('battery'),unauthorized_rejected=True,expired_rejected=True,physical_mode_rejected=True,
            measurements_under_normal_services=True,baseline_before_changes_available=False)
(root/'data/lab-acceptance.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps(report,ensure_ascii=False))
rclpy.shutdown();thread.join(timeout=2);node.destroy_node()
