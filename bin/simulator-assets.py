#!/usr/bin/env python3
"""Restore CAD data only, pinned by source revision and SHA; no vendor programs."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
volume = Path(os.environ.get("EXPLORER_SIM_VOLUME", "/Volumes/ExplorerSim"))
manifest = json.loads((ROOT / "config/arm_model.json").read_text())
folder = volume / "assets/m3_reference/meshes"
folder.mkdir(parents=True, exist_ok=True)
source = f'https://raw.githubusercontent.com/dimwael/yahboom_m3pro_description/{manifest["source_revision"]}/mjcf/m3pro/meshes/'
verified = {}
for relative, expected in manifest["asset_sha256"].items():
    if relative.endswith(".STL"):
        path = folder / Path(relative).name
        if not path.is_file():
            local = ROOT / relative
            content = local.read_bytes() if local.is_file() else urllib.request.urlopen(source+path.name, timeout=60).read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise SystemExit("Неверная контрольная сумма CAD: "+path.name)
            temporary = path.with_suffix(".download")
            temporary.write_bytes(content)
            temporary.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit("CAD повреждён: "+str(path))
        verified[path.name] = expected
(folder.parent / "manifest.json").write_text(json.dumps({"source": source, "sha256": verified}, indent=2))
print(f"Проверено {len(verified)} CAD-файлов")
