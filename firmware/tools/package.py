"""Create a pinned bench-install package after two matching target builds.

No device access. This is NOT hardware acceptance or authorization to write.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
CURRENT_SHA = 'e41e3cf0f8a406e354624cbf6ea15937e4da83d0ec0b9d04f76e16354b08b94c'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, why):
    if not condition:
        raise ValueError(why)


def ihex(data):
    result = []
    def record(address, kind, payload):
        raw = bytes([len(payload), address >> 8, address & 255, kind]) + payload
        result.append(':' + (raw+bytes([-sum(raw) & 255])).hex().upper())
    for offset in range(0, len(data), 16):
        address = 0x08000000 + offset
        if offset % 65536 == 0:
            record(0, 4, (address >> 16).to_bytes(2, 'big'))
        record(address & 65535, 0, data[offset:offset+16])
    record(0, 1, b'')
    return '\n'.join(result) + '\n'


def verify_elf(path):
    raw = path.read_bytes()
    require(raw[:7] == b'\x7fELF\x01\x01\x01', 'Need ARM ELF32 little-endian')
    header = struct.unpack_from('<16sHHIIIIIHHHHHH', raw)
    require(header[1] == 2 and header[2] == 40, 'Not an ARM executable')
    loads = []
    for i in range(header[10]):
        kind, offset, virtual, physical, size, memory_size, flags, align = struct.unpack_from(
            '<8I', raw, header[5]+i*header[9])
        if kind == 1:
            require(flags != 7, 'RWX load segment')
            if size:
                require(0x08000000 <= physical < physical+size <= 0x08020000,
                        'File data outside approved sector')
                require(offset+size <= len(raw), 'Truncated ELF segment')
            if memory_size and virtual >= 0x20000000:
                require(0x20000000 <= virtual < virtual+memory_size <= 0x20020000 or
                        0x30000000 <= virtual < virtual+memory_size <= 0x30000400,
                        'RAM outside DTCM / dedicated RGB DMA SRAM')
            loads.append(dict(address=hex(physical), virtual=hex(virtual), file_bytes=size,
                              memory_bytes=memory_size, flags=flags, alignment=align))
    require(len(loads) >= 2, 'Missing load segments')
    return loads


def run_tests(command, target):
    result = subprocess.run(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    target.write_text(result.stdout)
    require(result.returncode == 0, 'Tests failed: ' + str(target))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat-build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build = ROOT/'build-stm32'
    repeat = args.repeat_build
    comparisons = {}
    for name in ('explorer.bin', 'explorer.hex', 'explorer.elf'):
        comparisons[name] = dict(first=digest(build/name), second=digest(repeat/name))
        require(comparisons[name]['first'] == comparisons[name]['second'], 'Non-reproducible ' + name)
    source = (build/'generated/source.sha256').read_text().strip()
    require(source == (repeat/'generated/source.sha256').read_text().strip(), 'Source identity mismatch')
    segments = verify_elf(build/'explorer.elf')
    current = REPO/'vendor/completion-package-20260929/Explorer_completion/reference_only_NOT_AN_UPDATE/CURRENT_e41e3cf0.bin'
    require(digest(current) == CURRENT_SHA and current.stat().st_size == 1048576, 'CURRENT baseline mismatch')
    baseline = current.read_bytes()[:131072]
    app = (build/'explorer.bin').read_bytes()
    require(8 < len(app) < len(baseline), 'Application does not fit first sector')
    image = app + baseline[len(app):]
    destination = args.output.resolve()
    require(not destination.exists(), 'Output exists; do not overwrite a handed-off package')
    destination.mkdir(parents=True)
    for name in ('application', 'recovery', 'evidence', 'source', 'jetson-next-stage'):
        (destination/name).mkdir()
    for name in ('explorer.elf', 'explorer.map', 'explorer.bin', 'explorer.hex', 'build-result.json', 'SHA256SUMS'):
        shutil.copy2(build/name, destination/'application'/name)
    for source_file, name in ((build/'build.log', 'target-build.log'),
                              (ROOT/'build-host/build.log', 'host-build.log'),
                              (repeat/'build.log', 'repeat-target-build.log')):
        shutil.copy2(source_file, destination/'evidence'/name)
    run_tests([shutil.which('ctest'), '--test-dir', str(ROOT/'build-host'), '--output-on-failure'],
              destination/'evidence/host-tests.log')
    run_tests([sys.executable, '-m', 'unittest', 'discover', '-s', 'firmware/tests', '-p', 'test_install.py', '-v'],
              destination/'evidence/installer-tests.log')
    for name in ('core', 'src', 'include', 'vendor', 'tools', 'cmake', 'tests'):
        shutil.copytree(ROOT/name, destination/'source'/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('CMakeLists.txt', 'target.ld', 'sources.json', 'README.ru.md', 'INSTALL.ru.md', '.gitignore'):
        shutil.copy2(ROOT/name, destination/'source'/name)
    for name in ('controller_protocol.py', 'controller_feedback.py', 'controller_driver.py', 'timed_trajectory.py'):
        shutil.copy2(REPO/'src'/name, destination/'jetson-next-stage'/name)
    shutil.copy2(ROOT/'tools/install.py', destination/'install.py')
    shutil.copy2(ROOT/'INSTALL.ru.md', destination/'INSTALL.ru.md')
    (destination/'explorer-sector0.bin').write_bytes(image)
    (destination/'explorer-sector0.hex').write_text(ihex(image))
    (destination/'recovery/CURRENT-sector0.bin').write_bytes(baseline)
    image_sha = hashlib.sha256(image).hexdigest()
    baseline_sha = hashlib.sha256(baseline).hexdigest()
    (destination/'COMMANDS.txt').write_text(
        'cd "'+str(destination)+'"\n'
        'python3 install.py verify\npython3 install.py ports\n'
        'python3 install.py install --bench-ready --approve-sha256 '+image_sha+'\n\n'
        '# При нескольких устройствах добавьте --port /dev/cu.usbserial-...\n'
        '# Восстановление: замените sessions/ДАТА на папку конкретной установки.\n'
        'python3 install.py restore --backup "sessions/ДАТА" --bench-ready --approve-sha256 '+baseline_sha+'\n')
    (destination/'evidence/reproducibility.json').write_text(json.dumps(dict(
        source_sha256=source, comparisons=comparisons,
        map_note='MAP contains build-directory paths; supplied MAP is hashed, address bounds independently checked.',
        elf_load_segments=segments, hardware_accepted=False), indent=2)+'\n')
    manifest = dict(schema=1, version='0.2.0-bench1', status='bench_install_candidate',
                    hardware_accepted=False, source_sha256=source, image_sha256=image_sha,
                    current_full_dump_sha256=CURRENT_SHA, recovery_sector_sha256=baseline_sha,
                    chip=dict(device=0x450, revision=0x2003, flash_kib=[1024, 2048]),
                    jetson_runtime_integrated=False,
                    files={p.relative_to(destination).as_posix(): digest(p)
                           for p in sorted(destination.rglob('*')) if p.is_file()})
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    (destination/'SHA256SUMS').write_text(''.join(
        f'{digest(p)}  {p.relative_to(destination).as_posix()}\n'
        for p in sorted(destination.rglob('*')) if p.is_file()))
    result = subprocess.run([sys.executable, str(destination/'install.py'), 'verify'], capture_output=True, text=True)
    require(result.returncode == 0, result.stdout+result.stderr)
    archive = destination.with_suffix('.zip')
    require(not archive.exists(), 'Archive exists')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(destination.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(destination.parent))
    print(json.dumps(dict(directory=str(destination), archive=str(archive), image_sha256=image_sha,
                          archive_sha256=digest(archive), source_sha256=source), indent=2))


if __name__ == '__main__':
    main()
