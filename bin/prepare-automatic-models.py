#!/usr/bin/env python3
"""Idempotent local model setup in an isolated environment on ExplorerSim."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
VOLUME=Path('/Volumes/ExplorerSim');DATA=VOLUME/'ExplorerSim'
if not VOLUME.is_mount():raise SystemExit('Подключите диск ExplorerSim')
(DATA/'maintenance').mkdir(exist_ok=True)
with (DATA/'maintenance/model-install.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    python=VOLUME/'env-vision/bin/python';site=VOLUME/'env-vision/lib/python3.12/site-packages'
    if not python.is_file():subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(VOLUME/'env-vision')],check=True)
    site.mkdir(parents=True,exist_ok=True)
    (site/'explorer-base.pth').write_text(str(VOLUME/'env/lib/python3.12/site-packages')+'\n')
    probe=subprocess.run([str(python),'-c','import transformers,torch;assert transformers.__version__=="4.57.3"'],capture_output=True)
    if probe.returncode:
        subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check','transformers==4.57.3'],check=True)
    if sys.platform=='darwin':
        internal=Path.home()/'Library/Application Support/ExplorerSimulator/vision-runtime'
        if not (internal/'bin/python').is_file():
            subprocess.run([sys.executable,'-m','venv','--without-pip',str(internal)],check=True)
        (internal/'lib/python3.12/site-packages/explorer-vision.pth').write_text(str(site)+'\n'+str(VOLUME/'env/lib/python3.12/site-packages')+'\n')
    from huggingface_hub import snapshot_download
    expected=json.loads((ROOT/'config/grounding-model.json').read_text());target=DATA/'models/grounding-dino-tiny'
    valid=True
    for name,digest in expected['files'].items():
        try:
            with (target/name).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            valid=valid and actual==digest
        except OSError:valid=False
    if not valid:snapshot_download(expected['repository'],revision=expected['revision'],local_dir=target,allow_patterns=list(expected['files']),max_workers=2)
    for name,digest in expected['files'].items():
        with (target/name).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=digest:raise ValueError('Модель не прошла проверку: '+name)
    (target/'verified-install.json').write_text(json.dumps(dict(repository=expected['repository'],revision=expected['revision'],files=expected['files'],verified=time.time()),indent=2)+'\n')
    from simulator.language_learning import LocalLanguageModel
    planner=LocalLanguageModel(DATA)
    manifest=DATA/'llm-models/manifests/registry.ollama.ai/library/qwen3.5/4b'
    if not manifest.is_file():subprocess.run([sys.executable,str(ROOT/'bin/install-simulator-language.py')],check=True)
    print('Local language and vision models installed and verified')
