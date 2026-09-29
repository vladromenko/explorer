#!/usr/bin/env python3
"""Package two reproducible full firmware builds; never opens hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

from preflight import (BASE, CLI, HOST_TEST_EXECUTABLES, INSTALLED_SHA, SECTOR_BYTES, elf_segments, ihex_bytes,
                       native_write_command, require, sha, source_sha256,
                       verify_package, verify_vectors, write_json)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ('explorer.elf', 'explorer.map', 'explorer.bin', 'explorer.hex')


def ihex(data):
    records = []
    def record(address, kind, payload):
        raw = bytes((len(payload), address >> 8, address & 255, kind)) + payload
        records.append(':' + (raw + bytes((-sum(raw) & 255,))).hex().upper())
    for offset in range(0, len(data), 16):
        address = BASE + offset
        if offset % 65536 == 0:
            record(0, 4, (address >> 16).to_bytes(2, 'big'))
        record(address & 65535, 0, data[offset:offset + 16])
    record(0, 1, b'')
    return '\n'.join(records) + '\n'


def verify_builds(first, repeat, source_root):
    require(first.resolve() != repeat.resolve(), 'Нужны две отдельные сборки')
    source = source_sha256(source_root)
    comparisons = {}
    for build in (first, repeat):
        require((build / 'generated/source.sha256').read_text().strip() == source,
                'Исходники изменены после configure; нужна новая чистая сборка: ' + str(build))
        result = json.loads((build / 'build-result.json').read_text())
        require(result['source_sha256'] == source and result['flash_permitted'] is False,
                'Идентичность сборки не совпала')
        for name in ARTIFACTS:
            require(result['artifacts'][name] == sha((build / name).read_bytes()),
                    'Артефакт изменён после сборки: ' + name)
    for name in ARTIFACTS:
        comparisons[name] = sha((first / name).read_bytes()) == sha((repeat / name).read_bytes())
        require(comparisons[name], 'Повторная сборка не совпала побайтово: ' + name)
    a, b = (json.loads((build / 'build-result.json').read_text()) for build in (first, repeat))
    require(a['compiler_sha256'] == b['compiler_sha256'], 'Разные компиляторы в повторных сборках')
    application = (first / 'explorer.bin').read_bytes()
    verify_vectors(application)
    segments = elf_segments((first / 'explorer.elf').read_bytes(), application)
    require(ihex_bytes((first / 'explorer.hex').read_text()) == application, 'HEX отличается от BIN')
    require(bytes.fromhex(source) in application, 'Встроенная идентичность исходников не совпала')
    return source, comparisons, segments, a['compiler_sha256']


def archive_directory(directory, archive):
    require(not archive.exists(), 'Архив существует; выпуск не перезаписывается')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                output.write(path, path.relative_to(directory.parent))


def verify_host_build(host_build, source):
    record = json.loads((host_build / 'build-result.json').read_text())
    require(record.get('status') == 'host_built_and_tested' and record.get('source_sha256') == source,
            'Host-тесты собраны для других исходников; запустите build.py заново')
    require(set(record['tests']) == set(HOST_TEST_EXECUTABLES) and
            set(record['test_executables']) == set(HOST_TEST_EXECUTABLES.values()),
            'Нужны все пять групп: core, runtime, servo_bus, sensors, uart_io')
    for name, digest in record['test_executables'].items():
        require(sha((host_build / name).read_bytes()) == digest, 'Изменён исполняемый host-тест: ' + name)
    return record


def verified_test_count(returncode, output):
    summary = re.search(r'^100% tests passed(?:, 0 tests failed)? out of (\d+)\s*$', output, re.MULTILINE)
    executed = re.findall(r'Test\s+#\d+:\s+(\w+)\s+\.+\s+Passed', output)
    require(returncode == 0 and summary is not None and
            int(summary.group(1)) == len(HOST_TEST_EXECUTABLES) and
            len(executed) == len(HOST_TEST_EXECUTABLES) and set(executed) == set(HOST_TEST_EXECUTABLES),
            'Компьютерные тесты не пройдены:\n' + output)
    return len(executed)


def package(first, repeat, host_build, destination, current_sector, version,
            identity=None, port='/dev/cu.usbserial-02E0E664', source_root=ROOT):
    source, comparisons, segments, compiler = verify_builds(first, repeat, source_root)
    baseline = current_sector.read_bytes()
    require(len(baseline) == SECTOR_BYTES and sha(baseline) == INSTALLED_SHA,
            'Нужен точный установленный сектор a69d…, а не исторический ORIGINAL/CURRENT')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}', version) is not None, 'Недопустимая версия')
    archive = destination.with_name(destination.name + '.zip')
    require(not destination.exists() and not archive.exists(),
            'Не перезаписываем уже переданный выпуск')
    measured = dict(device=0x450, revision=0x2003, flash_kib=1024,
                    source='prior installed firmware HELLO; fresh ROM check before write')
    require(identity is not None, 'Нужен сохранённый HELLO с UID конкретной платы')
    require(all(identity.get(k) == measured[k] for k in ('device', 'revision', 'flash_kib')),
            'Сохранённый HELLO относится к другому MCU/ревизии/объёму flash')
    measured.update(identity)
    require(re.fullmatch('[0-9a-f]{24}', measured.get('uid', '')) is not None, 'UID должен содержать 12 байт')
    require(shutil.which('ctest'), 'ctest нужен для повторной проверки компьютерных тестов')
    host_record = verify_host_build(host_build, source)
    test_result = subprocess.run(['ctest', '--test-dir', str(host_build), '--output-on-failure'],
                                 capture_output=True, text=True)
    test_output = test_result.stdout + test_result.stderr
    tests_passed = verified_test_count(test_result.returncode, test_output)
    require(verify_host_build(host_build, source) == host_record,
            'Host-бинарники изменились во время проверки')
    # Read-only checks all precede creating the immutable release directory.
    destination.mkdir(parents=True)
    for name in ('application', 'recovery', 'evidence', 'jetson-staged'):
        (destination / name).mkdir()
    for name in ARTIFACTS + ('build-result.json', 'SHA256SUMS'):
        shutil.copy2(first / name, destination / 'application' / name)
    for path, name in ((first / 'build.log', 'first-build.log'),
                       (repeat / 'build.log', 'repeat-build.log'),
                       (host_build / 'build.log', 'host-build.log')):
        shutil.copy2(path, destination / 'evidence' / name)
    (destination / 'evidence/host-tests.log').write_text(test_output)
    write_json(destination / 'evidence/host-build-result.json', host_record)
    shutil.copytree(source_root, destination / 'source',
                    ignore=shutil.ignore_patterns('build-*', '__pycache__', '*.pyc', '.DS_Store'))
    require(source_sha256(destination / 'source') == source,
            'Исходники изменились во время упаковки; пакет не готов')
    shutil.copy2(source_root / 'tools/preflight.py', destination / 'preflight.py')
    shutil.copy2(source_root / 'INSTALL.ru.md', destination / 'INSTALL.ru.md')
    # A complete host source snapshot is staged only, never deployed or enabled here.
    repo = source_root.parent
    shutil.copytree(repo / 'src', destination / 'jetson-staged/src',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store'))
    shutil.copytree(repo / 'tests', destination / 'jetson-staged/tests',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store'))
    write_json(destination / 'jetson-staged/controller-profile.patch.json', dict(
        firmware_source_sha256=source, telemetry_only=True, hardware_accepted=False,
        blocking_reason_ru='Новый образ: до аппаратной приёмки включено только наблюдение.'))
    (destination / 'jetson-staged/README.ru.md').write_text(
        'Снимок совместимого кода Jetson; эти файлы автоматически не устанавливаются.\n'
        'После записи STM32 объединить controller-profile.patch.json с существующим\n'
        '/home/vlad/Explorer/config/controller-profile.json, сохранив device, transport\n'
        'и calibration_sha256. Поле firmware_source_sha256 сверяется с HELLO.\n'
        'Сначала проверить телеметрию и приёмку; telemetry_only=true и\n'
        'hardware_accepted=false в этом пакете не являются результатами испытаний.\n'
        'Запись профиля не заменяет приёмку и не должна автоматически разрешать движение.\n')
    application = (first / 'explorer.bin').read_bytes()
    image = application + baseline[len(application):]
    (destination / 'explorer-sector0.bin').write_bytes(image)
    (destination / 'explorer-sector0.hex').write_text(ihex(image))
    (destination / 'recovery/installed-sector0.bin').write_bytes(baseline)
    image_sha = sha(image)
    (destination / 'COMMANDS.txt').write_text(
        '# Выполнять из папки распакованного выпуска. Сначала согласовать этот SHA256:\n'
        '# ' + image_sha + '\n'
        'python3 preflight.py verify\n\n'
        '# Одно чтение первого сектора; никаких записей или сбросов. Новая папка сеанса:\n'
        'python3 preflight.py read --port ' + port + ' --output sessions/before-first-write\n\n'
        '# Только после согласования SHA256 и физической подготовки стенда — одна запись:\n'
        + native_write_command(CLI, port, 'explorer-sector0.bin') + '\n\n'
        '# Восстановление точно установленного ранее состояния (SHA256 ' + INSTALLED_SHA + '):\n'
        + native_write_command(CLI, port, 'recovery/installed-sector0.bin') + '\n\n'
        '# Приложение не запускается автоматически. BOOT отпустить; RESET — вручную на стенде.\n')
    manifest = dict(schema=2, version=version, status='built_not_installed_not_hardware_accepted',
                    flash_permitted=False, hardware_accepted=False, hardware_tested=False,
                    image_sha256=image_sha, recovery_sha256=INSTALLED_SHA,
                    source_sha256=source, compiler_sha256=compiler, measured_identity=measured,
                    application_bytes=len(application), sector_bytes=SECTOR_BYTES,
                    first_sector_address=hex(BASE), untouched_outside_sector0=True,
                    flash_banks=[dict(address='0x08000000', bytes=524288),
                                 dict(address='0x08100000', bytes=524288)],
                    elf_load_segments=segments, repeated_build_byte_identical=comparisons,
                    computer_tests_passed=tests_passed,
                    physical_tests_performed=[], jetson_deployed=False)
    manifest['files'] = {path.relative_to(destination).as_posix(): sha(path.read_bytes())
                         for path in sorted(destination.rglob('*')) if path.is_file()}
    write_json(destination / 'manifest.json', manifest)
    (destination / 'SHA256SUMS').write_text(''.join(
        sha(path.read_bytes()) + '  ' + path.relative_to(destination).as_posix() + '\n'
        for path in sorted(destination.rglob('*')) if path.is_file() and path.name != 'SHA256SUMS'))
    verify_package(destination)
    archive_directory(destination, archive)
    return dict(directory=str(destination), archive=str(archive),
                image_sha256=image_sha, source_sha256=source,
                archive_sha256=sha(archive.read_bytes()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-build', type=Path, required=True, help='First build-stm32 directory')
    parser.add_argument('--repeat-build', type=Path, required=True)
    parser.add_argument('--host-build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--current-sector', type=Path, default=ROOT.parent /
                        'releases/Explorer-STM32-0.2.0-bench3/explorer-sector0.bin')
    parser.add_argument('--identity-json', type=Path, required=True)
    parser.add_argument('--port', default='/dev/cu.usbserial-02E0E664')
    args = parser.parse_args()
    identity = json.loads(args.identity_json.read_text()) if args.identity_json else None
    print(json.dumps(package(args.first_build.resolve(), args.repeat_build.resolve(),
                             args.host_build.resolve(), args.output.resolve(), args.current_sector,
                             args.version, identity, args.port), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit('ПАКЕТ НЕ ГОТОВ: ' + str(error))
