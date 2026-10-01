"""Select a staged, exact MCU identity without granting movement or flashing.

The operator stages a release on Jetson before physically replacing firmware.
Only the driver owning the UART applies the profile after a matching HELLO.
Unknown boards/images and altered calibrations cannot use this transition.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(descriptor,'w') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def _hash(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def _board(identity):
    result={key:identity.get(key) for key in ('uid','device','revision','flash_kib')}
    if (not isinstance(result['uid'],str) or not re.fullmatch('[0-9a-f]{24}',result['uid']) or
        result['device']!=0x450 or result['revision']!=0x2003 or result['flash_kib']!=1024):
        raise ValueError('Контроллер не соответствует подтверждённой плате Explorer')
    return result


def stage_release(root, release, image_sha):
    """Read a prepared package and add its identity; active profile is untouched."""
    root,release=Path(root).resolve(),Path(release).resolve()
    manifest=json.loads((release/'manifest.json').read_text())
    if (manifest.get('schema')!=2 or not _hash(image_sha) or
        manifest.get('image_sha256')!=image_sha or
        not _hash(manifest.get('source_sha256')) or manifest.get('sector_bytes')!=131072 or
        manifest.get('first_sector_address')!='0x8000000'):
        raise ValueError('Не совпали конкретный образ или формат пакета')
    required={'explorer-sector0.bin','application/explorer.bin','recovery/installed-sector0.bin'}
    files=manifest.get('files',{})
    if not required.issubset(files):raise ValueError('Пакет неполон')
    for name,expected in files.items():
        path=(release/name).resolve()
        if not path.is_relative_to(release) or not _hash(expected) or digest(path)!=expected:
            raise ValueError('Артефакт пакета изменён: '+name)
    image=release/'explorer-sector0.bin'
    if image.stat().st_size!=131072 or digest(image)!=image_sha:
        raise ValueError('Хэш устанавливаемого сектора не совпал')
    source=manifest['source_sha256']
    if bytes.fromhex(source) not in (release/'application/explorer.bin').read_bytes():
        raise ValueError('Идентичность исходников отсутствует в образе')
    board=_board(manifest['measured_identity'])
    profile_path=root/'config/controller-profile.json'
    profile=json.loads(profile_path.read_text())
    calibration=digest(root/'config/controller-calibration.json')
    if calibration!=profile.get('calibration_sha256'):
        raise ValueError('Калибровка изменена относительно действующего профиля')
    state=json.loads((root/'data/controller-state.json').read_text())
    actual=state.get('identity') or {}
    if (not 0<=time.time()-state.get('at',0)<=3 or not state.get('telemetry_fresh') or
        _board(actual)!=board or actual.get('source_sha256')!=profile.get('firmware_source_sha256')):
        raise ValueError('Нет свежей идентичности текущего контроллера для подготовки перехода')
    registry_path=root/'config/controller-releases.json'
    registry=json.loads(registry_path.read_text()) if registry_path.exists() else dict(schema=1,entries={})
    if registry.get('schema')!=1 or not isinstance(registry.get('entries'),dict):
        raise ValueError('Неверный список подготовленных прошивок')
    common=dict(board=board,calibration_sha256=calibration)
    registry['entries'][profile['firmware_source_sha256']]=dict(common,
        version='previous-installed',reason='Возвращён прежний образ; необходима проверка контроллера.')
    registry['entries'][source]=dict(common,version=manifest['version'],image_sha256=image_sha,
        reason='Новый контроллер распознан. Следующий этап — физическая приёмка шасси и руки.')
    registry['staged_source_sha256']=source
    atomic_json(registry_path,registry)
    return dict(staged=True,version=manifest['version'],source_sha256=source,
                active_profile_changed=False,flash_performed=False,motion_enabled=False)


def select_controller_profile(root, profile, identity):
    root=Path(root);source=identity.get('source_sha256')
    if not _hash(source):raise ValueError('Нет точной идентичности прошивки STM32')
    expected_board=profile.get('controller_board')
    if expected_board is not None and _board(identity)!=expected_board:
        raise ValueError('Подключена другая плата STM32')
    if source==profile.get('firmware_source_sha256'):
        registry_path=root/'config/controller-releases.json'
        if registry_path.exists():
            entry=json.loads(registry_path.read_text()).get('entries',{}).get(source)
            if entry and _board(identity)!=entry.get('board'):
                raise ValueError('UID текущего образа не совпадает с подготовленной платой')
        return profile
    registry=json.loads((root/'config/controller-releases.json').read_text())
    if registry.get('schema')!=1:raise ValueError('Неизвестный формат подготовленных прошивок')
    entry=registry.get('entries',{}).get(source)
    if not entry or _board(identity)!=entry.get('board'):
        raise ValueError('Эта прошивка или плата не подготовлена для автоматического подключения')
    calibration=digest(root/'config/controller-calibration.json')
    if calibration!=entry.get('calibration_sha256') or calibration!=profile.get('calibration_sha256'):
        raise ValueError('Калибровка изменена после подготовки выпуска')
    # Persisted profile must still be the one this driver is replacing. Never
    # overwrite a concurrent operator/configuration change with cached data.
    path=root/'config/controller-profile.json'
    if json.loads(path.read_text())!=profile:
        raise ValueError('Профиль изменился параллельно; перезапустите драйвер')
    replacement=dict(profile,firmware_source_sha256=source,controller_board=entry['board'],
        telemetry_only=True,hardware_accepted=False,blocking_reason_ru=entry['reason'],
        controller_release=entry['version'])
    directory=root/'data/controller-profile-history';directory.mkdir(parents=True,exist_ok=True)
    atomic_json(directory/(str(time.time_ns())+'.json'),dict(previous=profile,selected=replacement,identity=identity))
    atomic_json(path,replacement)
    return replacement
