#!/usr/bin/env python3
"""CLI client for the persistent autonomy supervisor."""
import json
from pathlib import Path
import sys
import urllib.request
import uuid

ROOT=Path('/home/vlad/Explorer');BASE='http://127.0.0.1:8080/api/'


def request(path,value=None):
    token=(ROOT/'config/access_token').read_text().strip();data=None if value is None else json.dumps(value).encode()
    req=urllib.request.Request(BASE+path,data=data,headers={'Authorization':'Bearer '+token,
        **({'Content-Type':'application/json'} if data else {})})
    with urllib.request.urlopen(req,timeout=10) as response:return json.load(response)


def main():
    command=sys.argv[1] if len(sys.argv)>1 else 'status'
    if command=='status':result=request('autonomy')
    elif command=='start':
        arguments=sys.argv[2:];object_query='';destination='';goal_parts=[];index=0
        while index<len(arguments):
            if arguments[index] in ('--object','--destination'):
                if index+1>=len(arguments):raise ValueError('После '+arguments[index]+' нужно значение')
                if arguments[index]=='--object':object_query=arguments[index+1]
                else:destination=arguments[index+1]
                index+=2
            else:goal_parts.append(arguments[index]);index+=1
        goal=' '.join(goal_parts).strip()
        if not goal or not object_query or not destination:
            raise ValueError('Usage: explorer autonomy start --object <предмет> --destination <место> <цель>')
        result=request('autonomy/jobs',dict(request_id=uuid.uuid4().hex,goal=goal,attempts=1,time_budget_s=900,
            episode_budget_s=900,permissions=['base_motion','arm_motion'],reset_profile=None,
            object_query=object_query,destination_name=destination))
    elif command=='cancel':
        if len(sys.argv)!=3:raise ValueError('Usage: explorer autonomy cancel <job-id>')
        result=request('autonomy/jobs/'+sys.argv[2]+'/cancel',{})
    elif command=='grant':
        result=request('autonomy/permissions',dict(scopes=['base_motion','arm_motion'],hours=1,
            area='operator-defined area',objects=[]))
    else:raise ValueError('Usage: explorer autonomy {status|grant|start --object <предмет> --destination <место> <цель>|cancel <job-id>}')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    try:main()
    except (OSError,ValueError) as exc:raise SystemExit(str(exc))
