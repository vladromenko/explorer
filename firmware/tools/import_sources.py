"""Import only the offline dependencies used by the target, retaining provenance.

Usage: python3 firmware/tools/import_sources.py BOARD_SAMPLES COMPLETION_PACKAGE
This never connects to hardware. Imported files are verified by sources.json.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    board, package = map(Path, sys.argv[1:])
    records = []

    def copy(source, target, origin):
        destination = ROOT / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        if target.startswith('core/') and destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError('Refusing to overwrite developed core source: ' + target)
        shutil.copyfile(source, destination)
        records.append(dict(path=target, origin=origin,
                            sha256=hashlib.sha256(source.read_bytes()).hexdigest()))

    base = board / 'Microros_Samples/Publisher_odom'
    for source in sorted((base / 'Drivers').rglob('*')):
        if source.is_file() and source.suffix in ('.h', '.c', '.txt', '.md'):
            rel = source.relative_to(base).as_posix()
            copy(source, 'vendor/' + rel, 'Board_Samples/Microros_Samples/Publisher_odom/' + rel)
    for name in ('Core/Src/tim.c', 'Core/Inc/tim.h', 'Core/Src/system_stm32h7xx.c',
                 'Core/Startup/startup_stm32h743vgtx.s', 'APP/app_motor.c', 'APP/app_motor.h'):
        copy(base / name, 'vendor/board/' + Path(name).name,
             'Board_Samples/Microros_Samples/Publisher_odom/' + name)
    for example, names in {
        'Read_IMU': ('Core/Src/spi.c', 'Core/Inc/spi.h', 'APP/app_icm20948.h'),
        'Adc': ('Core/Src/adc.c', 'Core/Inc/adc.h'),
    }.items():
        for name in names:
            rel = 'STM32_Samples/' + example + '/' + name
            copy(board / rel, 'vendor/board/' + Path(name).name, 'Board_Samples/' + rel)
    for source in sorted((package / 'draft_core').rglob('*')):
        if source.is_file() and source.suffix in ('.c', '.h'):
            rel = source.relative_to(package / 'draft_core').as_posix()
            copy(source, 'core/' + rel, 'Explorer_completion/draft_core/' + rel)
    for sample in ('Read_IMU', 'Adc'):
        source_root = board / 'STM32_Samples' / sample
        for source in sorted((source_root / 'Drivers/STM32H7xx_HAL_Driver').rglob('*')):
            if source.is_file() and source.suffix in ('.c', '.h'):
                rel = source.relative_to(source_root).as_posix()
                destination = ROOT / 'vendor' / rel
                if destination.exists():
                    if destination.read_bytes() != source.read_bytes():
                        raise ValueError('Incompatible HAL sources: ' + rel)
                else:
                    copy(source, 'vendor/' + rel, 'Board_Samples/STM32_Samples/' + sample + '/' + rel)
    for name in ('Core/Inc/stm32h7xx_hal_conf.h', 'Core/Src/main.c'):
        copy(board / 'STM32_Samples/Motor' / name,
             'vendor/board/' + Path(name).name, 'Board_Samples/STM32_Samples/Motor/' + name)
    previous = ROOT / 'sources.json'
    if previous.exists():
        records.extend(r for r in json.loads(previous.read_text()) if r['origin'].startswith('https://'))
    (ROOT / 'sources.json').write_text(json.dumps(records, indent=2) + '\n')
    print(f'Imported {len(records)} source files; no hardware access')


if __name__ == '__main__':
    main()
