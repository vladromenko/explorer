#!/usr/bin/env python3
"""Offline verification and read-only ROM-loader checks. Never erase/write/reset."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import struct
import subprocess

BASE = 0x08000000
SECTOR_BYTES = 128 * 1024
INSTALLED_SHA = 'a69d32fb7faa8e74eb6a39c2c048fd3bbd58b8af8f8d69eeb8764a3c81950fb1'
CLI = '/Applications/STMicroelectronics/STM32Cube/STM32CubeProgrammer/STM32CubeProgrammer.app/Contents/MacOs/bin/STM32_Programmer_CLI'
HOST_TEST_EXECUTABLES = {'core': 'core_tests', 'runtime': 'runtime_tests',
                         'servo_bus': 'servo_bus_tests', 'sensors': 'sensor_tests',
                         'uart_io': 'uart_io_tests'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def safe_file(root, name):
    path = (root / name).resolve()
    require(path != root.resolve() and root.resolve() in path.parents,
            'Недопустимый путь в пакете: ' + name)
    require(path.is_file(), 'Нет файла: ' + name)
    return path


def verify_vectors(application):
    require(0x298 <= len(application) <= SECTOR_BYTES, 'Приложение не помещается в первый сектор')
    stack, reset = struct.unpack_from('<II', application)
    require(stack == 0x20020000, 'Начальный стек не соответствует DTCM проекта')
    require(reset & 1 and BASE <= (reset & ~1) < BASE + len(application),
            'Reset_Handler не находится внутри приложения')
    # Cortex-M7 + the 150 H743 external IRQ entries (166 words).
    for index in range(1, 166):
        entry = struct.unpack_from('<I', application, index * 4)[0]
        require(entry == 0 or (entry & 1 and BASE <= (entry & ~1) < BASE + len(application)),
                'Вектор вне приложения: ' + str(index))
    return {'initial_stack': hex(stack), 'reset_handler': hex(reset)}


def elf_segments(raw, application):
    require(len(raw) >= 52 and raw[:7] == b'\x7fELF\x01\x01\x01', 'Нужен ELF32 little-endian')
    header = struct.unpack_from('<16sHHIIIIIHHHHHH', raw)
    require(header[1:3] == (2, 40) and header[9] == 32, 'Нужен исполняемый ARM ELF')
    require(header[4] & 1 and BASE <= (header[4] & ~1) < BASE + len(application),
            'ELF entry за пределами приложения')
    require(header[5] + header[9] * header[10] <= len(raw), 'Обрезана таблица сегментов ELF')
    segments, loaded = [], []
    for index in range(header[10]):
        kind, offset, virtual, physical, size, memory_size, flags, align = struct.unpack_from(
            '<8I', raw, header[5] + index * header[9])
        if kind == 1:
            require(size <= memory_size and flags & 7 != 7, 'Некорректный/RWX сегмент ELF')
            require(offset + size <= len(raw), 'Обрезан сегмент ELF')
            if size:
                require(BASE <= physical < physical + size <= BASE + len(application),
                        'Загружаемый сегмент за пределами приложения/сектора 0')
                require(raw[offset:offset + size] == application[physical - BASE:physical - BASE + size],
                        'BIN не совпадает с загружаемым сегментом ELF')
                require(all(physical + size <= left or physical >= right for left, right in loaded),
                        'Перекрывающиеся загружаемые сегменты ELF')
                loaded.append((physical, physical + size))
            if memory_size:
                valid = (BASE <= virtual < virtual + memory_size <= BASE + SECTOR_BYTES or
                         0x20000000 <= virtual < virtual + memory_size <= 0x20020000 or
                         0x30000000 <= virtual < virtual + memory_size <= 0x30000400)
                require(valid, 'RAM/flash сегмент вне карты памяти')
            segments.append(dict(address=hex(physical), virtual=hex(virtual), file_bytes=size,
                                 memory_bytes=memory_size, flags=flags, alignment=align))
    require(len(loaded) >= 1 and min(left for left, _ in loaded) == BASE and
            max(right for _, right in loaded) == BASE + len(application),
            'BIN/ELF имеют разные границы загрузки')
    return segments


def ihex_bytes(text):
    memory, upper, ended = {}, 0, False
    for line in text.splitlines():
        require(not ended and line.startswith(':'), 'Недопустимая запись HEX')
        raw = bytes.fromhex(line[1:])
        require(len(raw) >= 5 and len(raw) == raw[0] + 5 and sum(raw) & 255 == 0,
                'Длина/контрольная сумма HEX не совпала')
        length, address, kind = raw[0], int.from_bytes(raw[1:3], 'big'), raw[3]
        payload = raw[4:-1]
        if kind == 0:
            for delta, byte in enumerate(payload):
                absolute = upper + address + delta
                require(absolute not in memory, 'Перекрытие HEX')
                require(BASE <= absolute < BASE + SECTOR_BYTES, 'HEX выходит за сектор 0')
                memory[absolute] = byte
        elif kind == 4:
            require(length == 2 and address == 0, 'Неверная запись расширенного адреса HEX')
            upper = int.from_bytes(payload, 'big') << 16
        elif kind == 5:
            require(length == 4 and address == 0, 'Неверная точка входа HEX')
            require(BASE <= (int.from_bytes(payload, 'big') & ~1) < BASE + SECTOR_BYTES,
                    'Точка входа HEX вне сектора 0')
        elif kind == 1:
            require(length == 0 and address == 0, 'Неверное окончание HEX')
            ended = True
        else:
            raise ValueError('Неподдерживаемый тип HEX: ' + str(kind))
    require(ended and memory and min(memory) == BASE, 'Неполный HEX')
    require(len(memory) == max(memory) - BASE + 1, 'Пробел внутри HEX')
    return bytes(memory[address] for address in range(BASE, max(memory) + 1))


def source_sha256(root):
    digest = hashlib.sha256()
    for folder in ('core', 'src', 'include', 'vendor', 'tools', 'cmake', 'tests'):
        for path in sorted((root / folder).rglob('*')):
            if path.is_file() and path.suffix in ('.c', '.h', '.s', '.py', '.cmake'):
                digest.update(path.relative_to(root).as_posix().encode() + b'\0' + path.read_bytes())
    for name in ('CMakeLists.txt', 'target.ld', 'sources.json'):
        digest.update(name.encode() + b'\0' + (root / name).read_bytes())
    return digest.hexdigest()


def verify_package(root):
    root = root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    require(manifest.get('schema') == 2 and manifest.get('hardware_accepted') is False and
            manifest.get('flash_permitted') is False, 'Неожиданный формат/статус кандидата')
    identity = manifest['measured_identity']
    require(all(identity.get(key) == value for key, value in
                (('device', 0x450), ('revision', 0x2003), ('flash_kib', 1024))) and
            re.fullmatch('[0-9a-f]{24}', identity.get('uid', '')) is not None,
            'Нужна измеренная идентичность конкретной платы, включая UID')
    require(manifest['first_sector_address'] == hex(BASE) and manifest['sector_bytes'] == SECTOR_BYTES and
            manifest['flash_banks'] == [dict(address='0x08000000', bytes=524288),
                                       dict(address='0x08100000', bytes=524288)],
            'Карта flash отличается от подтверждённой платы')
    for name, digest in manifest['files'].items():
        require(sha(safe_file(root, name).read_bytes()) == digest, 'Не совпал SHA256: ' + name)
    application = safe_file(root, 'application/explorer.bin').read_bytes()
    image = safe_file(root, 'explorer-sector0.bin').read_bytes()
    recovery = safe_file(root, 'recovery/installed-sector0.bin').read_bytes()
    require(len(image) == len(recovery) == SECTOR_BYTES, 'Нужны полные 128 КиБ сектора 0')
    require(sha(recovery) == INSTALLED_SHA == manifest['recovery_sha256'],
            'Восстановление должно возвращать реально установленный образ a69d…')
    require(image == application + recovery[len(application):], 'Изменён хвост сектора 0')
    require(sha(image) == manifest['image_sha256'], 'Не совпал SHA образа')
    verify_vectors(application)
    segments = elf_segments(safe_file(root, 'application/explorer.elf').read_bytes(), application)
    require(segments == manifest['elf_load_segments'], 'Изменена карта ELF')
    require(ihex_bytes(safe_file(root, 'application/explorer.hex').read_text()) == application,
            'HEX приложения отличается от BIN')
    require(ihex_bytes(safe_file(root, 'explorer-sector0.hex').read_text()) == image,
            'HEX сектора отличается от BIN')
    source = source_sha256(root / 'source')
    require(source == manifest['source_sha256'], 'Исходники отличаются от собранного образа')
    require(bytes.fromhex(source) in application, 'В BIN нет точной встроенной идентичности исходников')
    require(manifest['repeated_build_byte_identical'] == dict.fromkeys(
        ('explorer.elf', 'explorer.map', 'explorer.bin', 'explorer.hex'), True), 'Повторная сборка не совпала')
    return manifest


def option_values(text):
    def value(name):
        found = re.search(r'\b' + re.escape(name) + r'\s*:\s*0x([0-9a-fA-F]+)', text)
        require(found is not None, 'В выводе CubeProgrammer нет поля ' + name)
        return int(found.group(1), 16)
    expected = {'Chip ID': 0x450, 'RDP': 0xaa, 'SWAP_BANK': 0, 'IWDG1_SW': 1,
                'BOOT_CM7_ADD0': 0x800, 'nWRP0': 1}
    result = {key: value(key) for key in expected}
    require(result == expected, 'Идентификатор/защиты/загрузочный адрес не соответствуют плану')
    for prefix in ('PROT_AREA', 'SEC_AREA'):
        start, end = value(prefix + '_START1'), value(prefix + '_END1')
        require(start > end, 'В первом банке включена защищённая область')
        result[prefix + '_START1'], result[prefix + '_END1'] = start, end
    return result


def connection(executable, port):
    require(port.startswith('/dev/cu.usbserial-') or port.startswith('/dev/ttyUSB'),
            'Укажите подтверждённый UART-порт контроллера')
    return [str(executable), '-c', 'port=' + port, 'br=115200', 'P=EVEN', 'db=8', 'sb=1']


def native_write_command(executable, port, image):
    return shlex.join(connection(executable, port) + ['-d', str(image), '0x08000000', '-v'])


def read_only_preflight(root, port, output, executable=CLI):
    manifest = verify_package(root)
    require(not output.exists(), 'Папка сеанса уже существует; старые доказательства не перезаписываются')
    output.mkdir(parents=True)
    base = connection(executable, port)

    def run(args, label):
        # No generic passthrough: every command below is a fixed read operation.
        require(args == ['-ob', 'displ'] or (len(args) == 4 and args[0] == '-u'),
                'Допускаются только чтение и показ option bytes')
        command = base + args
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        text = result.stdout + result.stderr
        (output / (label + '.log')).write_text(shlex.join(command) + '\n' + text)
        clean = re.sub(r'\x1b\[[0-9;]*m', '', text)
        require(result.returncode == 0 and not re.search(r'\bError\s*:|Operation not permitted', clean),
                'Чтение не завершено; журнал: ' + str(output / (label + '.log')))
        return clean

    options = option_values(run(['-ob', 'displ'], 'options'))
    for address, size, filename in ((0x1ff1e880, 2, 'flashsize.bin'), (0x1ff1e800, 12, 'uid.bin'),
                                     (BASE, SECTOR_BYTES, 'before-sector0.bin')):
        path = output / filename
        run(['-u', hex(address), hex(size), str(path)], filename)
        require(path.is_file() and path.stat().st_size == size, 'Неполное чтение: ' + filename)
    flash_kib = struct.unpack('<H', (output / 'flashsize.bin').read_bytes())[0]
    require(flash_kib == 1024, 'Реальный размер flash отличается от подтверждённых 1024 КиБ')
    uid = (output / 'uid.bin').read_bytes().hex()
    saved_uid = manifest['measured_identity']['uid']
    require(saved_uid == uid, 'UID отличается от сохранённой платы')
    before = (output / 'before-sector0.bin').read_bytes()
    require(sha(before) == INSTALLED_SHA, 'Текущий сектор отличается от a69d…; запись не разрешена')
    result = dict(status='read_only_preflight_passed', recorded_at=datetime.now(timezone.utc).isoformat(),
                  uid=uid, flash_kib=flash_kib, options=options, before_sha256=sha(before),
                  candidate_sha256=manifest['image_sha256'], revision_live_read=False,
                  revision_from_prior_runtime=manifest['measured_identity']['revision'],
                  flash_performed=False, hardware_accepted=False)
    write_json(output / 'preflight.json', result)
    print('Текущий сектор a69d… сохранён один раз и совпал; flash=1024 КиБ. Записи/сброса не было.')
    print('Для записи всё ещё требуются согласование хэша и подготовленный стенд.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'read'))
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--port')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--programmer', default=CLI)
    args = parser.parse_args()
    if args.action == 'verify':
        manifest = verify_package(args.root)
        print('Файлы, ELF/BIN/HEX, векторы, исходники и повторная сборка проверены; устройство не открывалось.')
        print('SHA256 устанавливаемого сектора:', manifest['image_sha256'])
    else:
        require(args.port is not None and args.output is not None, 'Для чтения нужны --port и новая --output')
        read_only_preflight(args.root.resolve(), args.port, args.output.resolve(), args.programmer)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        raise SystemExit('ОСТАНОВЛЕНО без записи: ' + str(error))
