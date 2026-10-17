#!/usr/bin/env python3
"""Update only the desktop entry point; never stop or start a training job."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from simulator.desktop_launcher import render_launcher,DASHBOARD_URL


def main():
    if not Path('/Volumes/ExplorerSim').is_mount():raise ValueError('Подключите диск ExplorerSim')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('Сохраните проверенный код в Git перед установкой ярлыка')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    support=Path.home()/'Library/Application Support/ExplorerSimulator'
    source=support/('desktop-source-'+revision[:12])
    if not source.exists():
        content=subprocess.check_output(['git','archive',revision,'simulator','src','config','bin','docs'],cwd=ROOT)
        with tempfile.TemporaryDirectory(prefix='desktop-stage-',dir=support) as temporary:
            stage=Path(temporary)/'source';stage.mkdir()
            with tarfile.open(fileobj=io.BytesIO(content)) as archive:
                archive.extractall(stage,filter='data')
            files={str(p.relative_to(stage)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in stage.rglob('*') if p.is_file()}
            (stage/'source-manifest.json').write_text(json.dumps(dict(revision=revision,files=files)))
            stage.rename(source)
    launcher=Path.home()/'Applications/Explorer Simulator.app/Contents/MacOS/launch'
    backup=support/('launcher-before-neural-'+str(time.time_ns()))
    if launcher.exists():shutil.copy2(launcher,backup)
    script=render_launcher(source,support)
    temporary=launcher.with_suffix('.tmp');temporary.write_text(script);temporary.chmod(0o755)
    subprocess.run(['/bin/bash','-n',str(temporary)],check=True)
    temporary.replace(launcher)
    state=dict(revision=revision,source=str(source),url=DASHBOARD_URL,backup=str(backup),
               training_restarted=False,at=time.time())
    (support/'neural-launch.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
    print(json.dumps(state,ensure_ascii=False))


if __name__=='__main__':main()
