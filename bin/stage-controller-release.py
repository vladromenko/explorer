#!/usr/bin/env python3
"""Prepare exact image recognition on Jetson. No serial access or motion."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from controller_release import stage_release

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--image-sha256',required=True)
    args=p.parse_args()
    print(json.dumps(stage_release(ROOT,args.release,args.image_sha256),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
