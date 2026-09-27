#!/usr/bin/env python3
"""Operator-declared charger interlock; NOT an electrical charge measurement."""
import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('state',choices=['connected','disconnected','unknown'])
a=p.parse_args()
root=Path('/home/vlad/Explorer/data')
path=root/'charging-interlock.tmp'
with path.open('w') as stream:
 json.dump(dict(connected={'connected':True,'disconnected':False,'unknown':None}[a.state],
                source='operator_declaration_not_sensor',at=time.time()),stream)
 stream.flush();os.fsync(stream.fileno())
path.replace(root/'charging-interlock.json')
print('Interlock updated. Releasing it never clears the latched motion stop.')
