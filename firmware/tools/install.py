#!/usr/bin/env python3
"""One approved sector, verified preimage, no implicit reset or option writes.

Commands that touch a device run ONLY when the operator executes install/restore.
verify and ports do not open a device. See INSTALL.ru.md before installation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

BASE = 0x08000000
BANK2 = 0x08100000
SECTOR = 0x20000
CHUNK = 0x4000
CLI = Path('/Applications/STMicroelectronics/STM32Cube/STM32CubeProgrammer/'
           'STM32CubeProgrammer.app/Contents/MacOs/bin/STM32_Programmer_CLI')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def check_package(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    require(manifest['schema'] == 1 and manifest['hardware_accepted'] is False,
            'Неподдерживаемый формат пакета')
    for name, digest in manifest['files'].items():
        file = (root / name).resolve()
        require(file.is_relative_to(root.resolve()), 'Недопустимый путь в пакете')
        require(file.is_file() and sha(file.read_bytes()) == digest,
                'Файл отсутствует или SHA256 не совпал: ' + name)
    image = (root / 'explorer-sector0.bin').read_bytes()
    application = (root / 'application/explorer.bin').read_bytes()
    baseline = (root / 'recovery/CURRENT-sector0.bin').read_bytes()
    require(len(image) == len(baseline) == SECTOR and 8 < len(application) < SECTOR,
            'Неверный размер образа')
    require(image[:len(application)] == application and image[len(application):] == baseline[len(application):],
            'Образ не соответствует приложению и сохраняемому хвосту сектора')
    stack, entry = struct.unpack_from('<II', image)
    require(stack == 0x20020000 and entry & 1 and BASE <= (entry & ~1) < BASE + len(application),
            'Некорректная таблица векторов Cortex-M7')
    require(manifest['image_sha256'] == sha(image), 'Неверный хэш утверждаемого образа')
    return manifest, image, baseline


def chip_description(idcode, flash_kib, uid, registers):
    require(len(idcode) == 4 and len(flash_kib) == 4 and len(uid) == 12 and len(registers) == 40,
            'Неполное чтение идентификаторов/регистров')
    code = struct.unpack('<I', idcode)[0]
    device, revision = code & 0xfff, code >> 16
    kib = struct.unpack_from('<H', flash_kib)[0]
    optsr, prar, scar, wrp, boot = (struct.unpack_from('<I', registers, i)[0] for i in (0, 12, 20, 28, 36))
    require(device == 0x450 and revision == 0x2003,
            f'Этот образ рассчитан на STM32H743 rev.V: получены ID={device:#x}, REV={revision:#x}')
    require(kib in (1024, 2048), f'Неподдерживаемый реальный размер flash: {kib} КиБ')
    require((optsr >> 8) & 255 == 0xaa, 'Включена защита чтения; ничего не разблокируем')
    require(not optsr & 0x80000000, 'SWAP_BANK включён; этот план установки неприменим')
    require(optsr & 0x10, 'Аппаратный IWDG при старте требует отдельной процедуры')
    require((boot & 0xffff) == 0x800, 'BOOT_ADD0 не указывает на 0x08000000')
    require(wrp & 1, 'Первый сектор защищён от записи; защиту не меняем')
    for protection, name in ((prar, 'PCROP'), (scar, 'secure area')):
        require((protection & 0xfff) > ((protection >> 16) & 0xfff),
                'В первом банке настроена ' + name + '; нужен отдельный план')
    return dict(device=device, revision=revision, flash_kib=kib, uid=uid.hex(),
                bank_bytes=kib * 512, bank_addresses=[BASE, BANK2], sector_bytes=SECTOR,
                options=dict(optsr=optsr, prar=prar, scar=scar, wrp=wrp, boot=boot))


def chip_description_from_option_text(text, uid):
    """Validate identity/options exposed by the STM32 ROM UART loader.

    The H7 ROM loader reports DBGMCU_IDCODE while connecting but NACKs a
    direct upload from that peripheral address.  CubeProgrammer's option-byte
    display is the supported read-only path in bootloader mode.
    """
    def value(name, pattern=None):
        match = re.search(pattern or (r'\b' + re.escape(name) + r'\s*:\s*0x([0-9A-Fa-f]+)'), text)
        require(match is not None, 'В выводе CubeProgrammer отсутствует поле ' + name)
        return int(match.group(1), 16)

    device = value('Chip ID')
    size = re.search(r'\bNVM size\s*:\s*(\d+)\s*MBytes', text)
    require(size is not None, 'CubeProgrammer не сообщил размер flash')
    kib = int(size.group(1)) * 1024
    require(device == 0x450, f'Этот образ рассчитан на STM32H743/H753: получен ID={device:#x}')
    require(kib in (1024, 2048), f'Неподдерживаемый реальный размер flash: {kib} КиБ')
    require(len(uid) == 12, 'Неполное чтение UID')
    require(value('RDP') == 0xaa, 'Включена защита чтения; ничего не разблокируем')
    require(value('SWAP_BANK') == 0, 'SWAP_BANK включён; этот план установки неприменим')
    require(value('IWDG1_SW') == 1, 'Аппаратный IWDG при старте требует отдельной процедуры')
    require(value('BOOT_CM7_ADD0') == 0x800, 'BOOT_CM7_ADD0 не указывает на 0x08000000')
    require(value('nWRP0') == 1, 'Первый сектор защищён от записи; защиту не меняем')
    for prefix, label in (('PROT_AREA', 'PCROP'), ('SEC_AREA', 'secure area')):
        start = value(prefix + '_START1')
        end = value(prefix + '_END1')
        require(start > end, 'В первом банке настроена ' + label + '; нужен отдельный план')
    options = dict(rdp=value('RDP'), swap_bank=value('SWAP_BANK'),
                   iwdg1_sw=value('IWDG1_SW'), boot_cm7_add0=value('BOOT_CM7_ADD0'),
                   nwrp0=value('nWRP0'), prot_area_start1=value('PROT_AREA_START1'),
                   prot_area_end1=value('PROT_AREA_END1'), sec_area_start1=value('SEC_AREA_START1'),
                   sec_area_end1=value('SEC_AREA_END1'))
    return dict(device=device, revision='verified_at_runtime', flash_kib=kib, uid=uid.hex(),
                bank_bytes=kib * 512, bank_addresses=[BASE, BANK2], sector_bytes=SECTOR,
                options=options)


class Programmer:
    def __init__(self, executable, port, directory):
        self.executable, self.port, self.directory = str(executable), port, directory
        self.number = 0

    def command(self, *args):
        self.number += 1
        command = [self.executable, '-c', 'port=' + self.port, 'br=115200',
                   'P=EVEN', 'db=8', 'sb=1', 'fc=OFF', *map(str, args)]
        log = self.directory / f'command-{self.number:04d}.log'
        with log.open('w') as output:
            output.write(json.dumps(command) + '\n')
            output.flush()
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=180)
        text = re.sub(r'\x1b\[[0-9;]*m', '', log.read_text(errors='replace'))
        require(result.returncode == 0 and not re.search(r'\bError\s*:|Error:|Operation not permitted', text),
                'STM32CubeProgrammer сообщил ошибку. Журнал: ' + str(log))
        return text

    def read(self, address, size, name):
        failures = []
        for attempt in range(1, 4):
            output = self.directory / (name if attempt == 1 else name + f'.attempt-{attempt}')
            require(not output.exists(), 'Отказ перезаписывать доказательства: ' + str(output))
            try:
                self.command('--upload', hex(address), hex(size), output)
                require(output.is_file() and output.stat().st_size == size,
                        'Неполное чтение: ' + str(output))
                return output.read_bytes()
            except (ValueError, subprocess.TimeoutExpired) as error:
                failures.append(str(error))
                print(f'Повтор безопасного чтения блока {address:#x}: {attempt}/3', flush=True)
        raise ValueError('Три чтения блока завершились ошибкой: ' + '; '.join(failures))

    def identify(self, label):
        # Direct reads of DBGMCU/FLASH registers are NACKed by the STM32H7
        # system UART bootloader.  The connection and `-ob displ` output are
        # read-only and contain the same installation-critical fields.
        option_text = self.command('-ob', 'displ')
        uid = self.read(0x1ff1e800, 12, label + '-uid.bin')
        return chip_description_from_option_text(option_text, uid)

    def read_bank(self, base, size, label):
        result = bytearray()
        for offset in range(0, size, CHUNK):
            result.extend(self.read(base + offset, min(CHUNK, size-offset), f'{label}-{offset:06x}.bin'))
            print(f'{label}: {len(result)//1024}/{size//1024} КиБ', flush=True)
        (self.directory / (label + '.bin')).write_bytes(result)
        return bytes(result)

    def program_sector(self, image):
        require(len(image) == SECTOR, 'Разрешён только один полный сектор')
        self.command('-e', '0')
        for offset in range(0, SECTOR, CHUNK):
            part = self.directory / f'write-{offset:06x}.bin'
            part.write_bytes(image[offset:offset+CHUNK])
            self.command('--skipErase', '--download', part, hex(BASE+offset), '-v')


def install(root, programmer, approved_sha, bench_ready):
    manifest, image, baseline = check_package(root)
    require(bench_ready, 'Нужно подтвердить стенд: колёса вывешены, рука поддержана')
    require(approved_sha == manifest['image_sha256'], 'Не согласован точный SHA256 устанавливаемого образа')
    print('Образ:', approved_sha, flush=True)
    print('Проверка платы и сохранение обоих банков перед стиранием…', flush=True)
    identity = programmer.identify('before')
    bank1 = programmer.read_bank(BASE, identity['bank_bytes'], 'before-bank1')
    bank2 = programmer.read_bank(BANK2, identity['bank_bytes'], 'before-bank2')
    require(bank1[:SECTOR] == baseline,
            'Текущий первый сектор отличается от сохранённого CURRENT. Запись не начата; бэкап сохранён')
    # A second fresh read of the affected sector also catches incomplete/unstable
    # serial reads and an intervening device reset/change before destructive work.
    again = programmer.read_bank(BASE, SECTOR, 'confirm-sector0')
    require(again == bank1[:SECTOR], 'Повторное чтение сектора не совпало; запись не начата')
    require(programmer.identify('prewrite') == identity, 'Плата или options изменились; запись не начата')
    backup = dict(schema=1, identity=identity, image_sha256=approved_sha,
                  bank1_sha256=sha(bank1), bank2_sha256=sha(bank2),
                  sector0_sha256=sha(bank1[:SECTOR]),
                  restore_only_sector0=True, stage='backup_verified')
    save_json(programmer.directory / 'backup.json', backup)
    print('Бэкап проверен. Запись только сектора 0…', flush=True)
    # The record exists BEFORE erase so recovery remains possible after a failure.
    save_json(programmer.directory / 'operation.json', dict(stage='writing', image_sha256=approved_sha))
    programmer.program_sector(image)
    after1 = programmer.read_bank(BASE, identity['bank_bytes'], 'after-bank1')
    after2 = programmer.read_bank(BANK2, identity['bank_bytes'], 'after-bank2')
    require(after1[:SECTOR] == image, 'Проверка записанного сектора не прошла; не запускайте приложение')
    require(after1[SECTOR:] == bank1[SECTOR:] and after2 == bank2, 'Изменилась область вне сектора 0; сохраните все журналы')
    require(programmer.identify('after') == identity, 'Изменились идентификаторы или option bytes')
    save_json(programmer.directory / 'operation.json', dict(stage='written_and_readback_verified',
              image_sha256=sha(after1[:SECTOR]), other_flash_unchanged=True,
              hardware_accepted=False, application_started=False))
    print('ЗАПИСЬ И ЧТЕНИЕ ПРОВЕРЕНЫ. Остальные секторы и options не изменились.')
    print('Автоматического запуска нет. Удерживая стенд подготовленным, отпустите BOOT и нажмите RESET.')
    print('Это начало стендовой приёмки, не подтверждение работоспособности приводов.')


def restore(programmer, saved, approved_sha, bench_ready):
    require(bench_ready, 'Подтвердите подготовку стенда')
    record = json.loads((saved / 'backup.json').read_text())
    bank1 = (saved / 'before-bank1.bin').read_bytes()
    bank2 = (saved / 'before-bank2.bin').read_bytes()
    require(sha(bank1) == record['bank1_sha256'] and sha(bank2) == record['bank2_sha256'], 'Повреждён бэкап')
    image = bank1[:SECTOR]
    require(len(image) == SECTOR and sha(image) == record['sector0_sha256'] == approved_sha,
            'Нужен точный SHA256 восстанавливаемого первого сектора')
    require(programmer.identify('restore-before') == record['identity'], 'Бэкап принадлежит другой плате или options изменились')
    current1 = programmer.read_bank(BASE, len(bank1), 'restore-before-bank1')
    current2 = programmer.read_bank(BANK2, len(bank2), 'restore-before-bank2')
    require(current1[SECTOR:] == bank1[SECTOR:] and current2 == bank2,
            'Есть отличия вне первого сектора; узкое восстановление неприменимо')
    save_json(programmer.directory / 'operation.json', dict(stage='restoring_sector0', source_backup=str(saved)))
    programmer.program_sector(image)
    after1 = programmer.read_bank(BASE, len(bank1), 'restore-after-bank1')
    after2 = programmer.read_bank(BANK2, len(bank2), 'restore-after-bank2')
    require(after1 == bank1 and after2 == bank2, 'Восстановление не прошло проверку чтением')
    require(programmer.identify('restore-after') == record['identity'], 'Изменились идентификаторы/options')
    save_json(programmer.directory / 'operation.json', dict(stage='restored_and_verified', source_backup=str(saved)))
    print('Восстановлено именно состояние перед этой установкой. Оба банка сверены. Автозапуска нет.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('verify', 'ports', 'install', 'restore'))
    parser.add_argument('--port')
    parser.add_argument('--programmer', type=Path, default=CLI)
    parser.add_argument('--approve-sha256')
    parser.add_argument('--bench-ready', action='store_true')
    parser.add_argument('--backup', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    ports = sorted(set(Path('/dev').glob('cu.usbserial-*')) | set(Path('/dev').glob('cu.SLAB_USBtoUART*')))
    if args.operation == 'ports':
        print('\n'.join(map(str, ports)) or 'Контроллер не найден. Подключите разъём данных платы к Mac.')
    elif args.operation == 'verify':
        manifest, _, _ = check_package(root)
        print('Файлы и таблица векторов проверены. Устройство не открывалось.')
        print('SHA256 образа:', manifest['image_sha256'])
    else:
        port = args.port or (str(ports[0]) if len(ports) == 1 else None)
        require(port and Path(port).exists(), 'Укажите единственный порт контроллера через --port')
        require(args.programmer.is_file(), 'STM32CubeProgrammer CLI не найден')
        require(args.bench_ready and args.approve_sha256, 'Нужны --bench-ready и --approve-sha256')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        directory = root / 'sessions' / stamp
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        print('Журналы и резервная копия:', directory, flush=True)
        programmer = Programmer(args.programmer, port, directory)
        if args.operation == 'install':
            install(root, programmer, args.approve_sha256, args.bench_ready)
        else:
            require(args.backup is not None, 'Для restore укажите --backup с папкой сессии установки')
            restore(programmer, args.backup.resolve(), args.approve_sha256, args.bench_ready)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        print('ОСТАНОВЛЕНО:', error, file=sys.stderr)
        print('Автоматических повторов/стирания других областей нет. Сохраните журналы sessions.', file=sys.stderr)
        sys.exit(1)
