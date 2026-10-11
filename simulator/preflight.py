"""Check the installed Mac learning runtime before spending a night on it."""
from importlib.metadata import version
from pathlib import Path
import hashlib
import shutil


def night_readiness(root):
    root = Path(root)
    blockers = []
    versions = {}
    for name, expected in (('lerobot','0.6.1'), ('torch','2.11.0'), ('torchvision','0.26.0')):
        try:
            versions[name] = version(name)
            if versions[name] != expected:
                blockers.append(name+': требуется '+expected)
        except Exception as exc:
            blockers.append(name+': '+str(exc))
    device = None
    try:
        from .learning import choose_device
        device = choose_device('auto')
        from lerobot.policies.act import ACTPolicy  # Verify the installed ACT import.
    except Exception as exc:
        blockers.append('Среда ACT: '+str(exc))
    free = shutil.disk_usage(root).free
    if free < 20*1024**3:
        blockers.append('Для ночного задания нужно минимум 20 ГБ свободного места')
    backbone = root.parent / 'cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
    backbone_ok = (backbone.is_file() and
        hashlib.sha256(backbone.read_bytes()).hexdigest().startswith('f37072fd'))
    if not backbone_ok:
        blockers.append('Нет проверенных весов ResNet18 в локальном кэше')
    return dict(ready=not blockers, blocked_by=blockers, versions=versions, device=device,
        free_gib=round(free/1024**3,1), backbone_cached=backbone_ok,
        recommended_attempts=100, recommended_steps=10000, hours=8,
        extra_model_download_required=not backbone_ok,
        physical_success_guaranteed=False)
