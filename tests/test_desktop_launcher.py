import json
import shlex
import subprocess

from simulator.desktop_launcher import DASHBOARD_URL,render_launcher


def run_launcher(tmp_path,healthy):
    calls=tmp_path/'calls.jsonl'
    def tool(name,body):
        path=tmp_path/name
        path.write_text('#!/usr/bin/env python3\nimport json,sys\nwith open('+repr(str(calls))+',"a") as f:f.write(json.dumps([sys.argv[0],sys.argv[1:]])+"\\n")\n'+body+'\n')
        path.chmod(0o755)
        return shlex.quote(str(path))
    replacements={
        '/usr/bin/curl':tool('health','sys.exit('+str(0 if healthy else 7)+')'),
        '/usr/bin/open':tool('open','sys.exit(0)'),
        '/sbin/mount':tool('mount','sys.exit(0)'),
        '/usr/bin/osascript':tool('alert','sys.exit(0)'),
    }
    # A missing source guarantees these cases cannot accidentally start Python.
    script=render_launcher(tmp_path/'missing-source',tmp_path/'support')
    for original,replacement in replacements.items():script=script.replace(original,replacement)
    subprocess.run(['/bin/bash','-n'],input=script,text=True,check=True)
    subprocess.run(['/bin/bash'],input=script,text=True,check=True,timeout=5)
    return [(row[0].rsplit('/',1)[-1],row[1]) for row in map(json.loads,calls.read_text().splitlines())]


def test_live_training_opens_neural_dashboard_without_mount_or_process_start(tmp_path):
    calls=run_launcher(tmp_path,True)
    assert [name for name,_ in calls]==['health','open']
    assert calls[-1][1]==[DASHBOARD_URL]


def test_missing_disk_alerts_without_starting_a_service_or_opening_stale_ui(tmp_path):
    calls=run_launcher(tmp_path,False)
    assert [name for name,_ in calls]==['health','mount','alert']
