"""Clean host tests and/or full STM32 link. Never opens a serial port."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from preflight import HOST_TEST_EXECUTABLES, source_sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=('host', 'stm32', 'all'), default='all')
    parser.add_argument('--toolchain', type=Path)
    parser.add_argument('--output-root', type=Path, default=ROOT)
    args = parser.parse_args()
    cmake = shutil.which('cmake')
    if not cmake:
        raise SystemExit('cmake is required')
    environment = os.environ.copy()
    if args.target != 'host':
        directory = args.toolchain or Path(environment.get('ARM_GCC_BIN', ''))
        if not (directory / 'arm-none-eabi-gcc').exists():
            candidates = sorted(Path('/Applications/STM32CubeIDE.app/Contents/Eclipse/plugins').glob(
                'com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.*/tools/bin'))
            if len(candidates) != 1:
                raise SystemExit('Pass --toolchain /path/to/arm-none-eabi/bin')
            directory = candidates[0]
        environment['ARM_GCC_BIN'] = str(directory)

    def run(command, log):
        result = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log.write(result.stdout)
        print(result.stdout[-8000:])
        if result.returncode:
            raise SystemExit(result.returncode)

    for target in (('host', 'stm32') if args.target == 'all' else (args.target,)):
        source_before = source_sha256(ROOT)
        build = args.output_root.resolve() / ('build-' + target)
        build.mkdir(parents=True, exist_ok=True)
        with (build / 'build.log').open('w') as log:
            configure = [cmake, '-S', str(ROOT), '-B', str(build), '-DCMAKE_BUILD_TYPE=Debug']
            if target == 'stm32':
                configure += ['-DCMAKE_TOOLCHAIN_FILE=' + str(ROOT / 'cmake/arm-gcc.cmake')]
                run([str(directory / 'arm-none-eabi-gcc'), '--version'], log)
            run(configure, log)
            run([cmake, '--build', str(build), '--clean-first', '--parallel', '4'], log)
            if target == 'host':
                run([str(Path(cmake).with_name('ctest')), '--test-dir', str(build), '--output-on-failure'], log)
                if source_sha256(ROOT) != source_before:
                    raise SystemExit('Sources changed during host build/test; rebuild after edits stop')
                (build / 'build-result.json').write_text(json.dumps(dict(
                    status='host_built_and_tested', source_sha256=source_before,
                    source_root=str(ROOT.resolve()), tests=list(HOST_TEST_EXECUTABLES),
                    test_executables={name: hashlib.sha256((build/name).read_bytes()).hexdigest()
                                      for name in HOST_TEST_EXECUTABLES.values()},
                    flash_permitted=False), indent=2) + '\n')
            else:
                if source_sha256(ROOT) != source_before:
                    raise SystemExit('Sources changed during STM32 build; rebuild after edits stop')
                artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in build.glob('explorer.*') if p.suffix in ('.elf', '.map', '.bin', '.hex')}
                (build / 'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n, h in sorted(artifacts.items())))
                (build / 'build-result.json').write_text(json.dumps(dict(
                    status='built_not_hardware_accepted', artifacts=artifacts,
                    source_sha256=(build/'generated/source.sha256').read_text().strip(),
                    compiler_sha256=hashlib.sha256((directory/'arm-none-eabi-gcc').read_bytes()).hexdigest(),
                    flash_permitted=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
