"""Release checks exercise corruption and prohibit hardware writes in preflight."""
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1] / 'firmware/tools'
sys.path.insert(0, str(TOOLS))
import preflight
import package_release
sys.path.pop(0)

OPTION_TEXT = '''Chip ID: 0x450
NVM size: 2 MBytes (default)
RDP: 0xAA
SWAP_BANK: 0x0
IWDG1_SW: 0x1
BOOT_CM7_ADD0: 0x800
nWRP0: 0x1
PROT_AREA_START1: 0xFF
PROT_AREA_END1: 0x0
SEC_AREA_START1: 0xFF
SEC_AREA_END1: 0x0
'''

# Exact CTest result block from the actual 0.2.2-rc1 first host build on this Mac.
REAL_CTEST_OUTPUT = '''Test project /private/tmp/explorer-0.2.2-rc1-first/build-host
    Start 1: core
1/5 Test #1: core .............................   Passed    0.71 sec
    Start 2: runtime
2/5 Test #2: runtime ..........................   Passed    0.42 sec
    Start 3: servo_bus
3/5 Test #3: servo_bus ........................   Passed    0.38 sec
    Start 4: sensors
4/5 Test #4: sensors ..........................   Passed    0.42 sec
    Start 5: uart_io
5/5 Test #5: uart_io ..........................   Passed    0.38 sec

100% tests passed out of 5

Total Test time (real) =   2.32 sec
'''


def make_elf(application):
    header = struct.pack('<16sHHIIIIIHHHHHH', b'\x7fELF\x01\x01\x01' + b'\0' * 9,
                         2, 40, 1, preflight.BASE + 665, 52, 0, 0, 52, 32, 2, 0, 0, 0)
    code = struct.pack('<8I', 1, 116, preflight.BASE, preflight.BASE,
                       len(application), len(application), 5, 4)
    bss = struct.pack('<8I', 1, 116 + len(application), 0x20000000,
                      preflight.BASE + len(application), 0, 32, 6, 4)
    return header + code + bss + application


class FirmwareReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'firmware'
        for folder in ('core', 'src', 'include', 'vendor', 'tools', 'cmake', 'tests'):
            (self.source / folder).mkdir(parents=True)
        for name in ('CMakeLists.txt', 'target.ld', 'sources.json', 'INSTALL.ru.md'):
            (self.source / name).write_text('test fixture\n')
        (self.source / 'src/main.c').write_text('/* fixture */\n')
        shutil.copy2(TOOLS / 'preflight.py', self.source / 'tools/preflight.py')
        (self.root / 'src').mkdir()
        (self.root / 'src/controller_driver.py').write_text('# fixture\n')
        (self.root / 'tests').mkdir()
        self.source_digest = preflight.source_sha256(self.source)
        self.application = bytearray(768)
        struct.pack_into('<II', self.application, 0, 0x20020000, preflight.BASE + 665)
        self.application[700:732] = bytes.fromhex(self.source_digest)
        self.application = bytes(self.application)
        self.first, self.repeat, self.host = (self.root / n for n in ('first', 'repeat', 'host'))
        for build in (self.first, self.repeat):
            (build / 'generated').mkdir(parents=True)
            (build / 'generated/source.sha256').write_text(self.source_digest + '\n')
            (build / 'explorer.bin').write_bytes(self.application)
            (build / 'explorer.elf').write_bytes(make_elf(self.application))
            (build / 'explorer.hex').write_text(package_release.ihex(self.application))
            (build / 'explorer.map').write_text('fixture map\n')
            (build / 'build.log').write_text('fixture build\n')
            (build / 'SHA256SUMS').write_text('fixture sums\n')
            preflight.write_json(build / 'build-result.json', dict(
                source_sha256=self.source_digest, flash_permitted=False, compiler_sha256='a' * 64,
                artifacts={name: preflight.sha((build / name).read_bytes()) for name in package_release.ARTIFACTS}))
        self.host.mkdir()
        (self.host / 'build.log').write_text('fixture host build\n')
        for name in preflight.HOST_TEST_EXECUTABLES.values():
            (self.host / name).write_bytes(b'fixture test executable: ' + name.encode())
        preflight.write_json(self.host / 'build-result.json', dict(
            status='host_built_and_tested', source_sha256=self.source_digest,
            tests=list(preflight.HOST_TEST_EXECUTABLES),
            test_executables={name: preflight.sha((self.host / name).read_bytes())
                              for name in preflight.HOST_TEST_EXECUTABLES.values()}))
        self.baseline = bytes((i % 256 for i in range(preflight.SECTOR_BYTES)))
        self.baseline_path = self.root / 'installed.bin'
        self.baseline_path.write_bytes(self.baseline)
        self.baseline_sha = preflight.sha(self.baseline)
        self.uid = '36003c000551333031343239'
        self.identity = dict(device=0x450, revision=0x2003, flash_kib=1024, uid=self.uid)
        self.destination = self.root / 'Explorer-STM32-0.2.2-test'

    def tearDown(self):
        self.temp.cleanup()

    def package(self):
        log = ''.join(f'{i}/5 Test #{i}: {name} ... Passed 0.01 sec\n'
                      for i, name in enumerate(preflight.HOST_TEST_EXECUTABLES, 1))
        result = subprocess.CompletedProcess(['ctest'], 0, log +
                                             '100% tests passed, 0 tests failed out of 5\n', '')
        with mock.patch.object(package_release, 'INSTALLED_SHA', self.baseline_sha), \
             mock.patch.object(preflight, 'INSTALLED_SHA', self.baseline_sha), \
             mock.patch.object(package_release.subprocess, 'run', return_value=result), \
             mock.patch.object(package_release.shutil, 'which', return_value='/usr/bin/ctest'):
            return package_release.package(self.first, self.repeat, self.host, self.destination,
                                           self.baseline_path, '0.2.2-test', self.identity,
                                           source_root=self.source)

    def test_release_roundtrip_and_native_single_sector_commands(self):
        result = self.package()
        self.assertTrue(Path(result['archive']).is_file())
        self.assertEqual(Path(result['archive']).name, 'Explorer-STM32-0.2.2-test.zip')
        with mock.patch.object(preflight, 'INSTALLED_SHA', self.baseline_sha):
            manifest = preflight.verify_package(self.destination)
        self.assertFalse(manifest['hardware_accepted'])
        self.assertFalse(manifest['flash_permitted'])
        image = (self.destination / 'explorer-sector0.bin').read_bytes()
        self.assertEqual(image, self.application + self.baseline[len(self.application):])
        commands = (self.destination / 'COMMANDS.txt').read_text()
        writes = [line for line in commands.splitlines() if ' -d ' in line]
        self.assertEqual(len(writes), 2)  # one candidate command and one explicit recovery command
        self.assertTrue(all(line.endswith('0x08000000 -v') for line in writes))
        self.assertFalse(any(token in commands for token in (' -e ', ' --skipErase', ' -rst', ' -g ')))
        patch = json.loads((self.destination / 'jetson-staged/controller-profile.patch.json').read_text())
        self.assertTrue(patch['telemetry_only'])
        self.assertFalse(patch['hardware_accepted'])

    def test_real_ctest_summary_formats(self):
        self.assertEqual(package_release.verified_test_count(0, REAL_CTEST_OUTPUT), 5)
        previous = REAL_CTEST_OUTPUT.replace('100% tests passed out of 5',
                                             '100% tests passed, 0 tests failed out of 5')
        self.assertEqual(package_release.verified_test_count(0, previous), 5)

    def test_failed_or_incomplete_ctest_output_rejected(self):
        for code, text in ((1, REAL_CTEST_OUTPUT),
                           (0, REAL_CTEST_OUTPUT.replace('uart_io', 'other')),
                           (0, REAL_CTEST_OUTPUT.replace('100% tests passed', '80% tests passed')),
                           (0, REAL_CTEST_OUTPUT.replace('out of 5', 'out of 3')),
                           (0, REAL_CTEST_OUTPUT.replace('Passed    0.38 sec', 'Failed    0.38 sec'))):
            with self.subTest(code=code, output=text), self.assertRaisesRegex(ValueError, 'не пройдены'):
                package_release.verified_test_count(code, text)

    def test_source_edit_after_build_rejected(self):
        (self.source / 'src/main.c').write_text('/* changed after configure */\n')
        with self.assertRaisesRegex(ValueError, 'после configure'):
            package_release.verify_builds(self.first, self.repeat, self.source)

    def test_repeated_map_must_match_even_when_both_individual_hashes_valid(self):
        (self.repeat / 'explorer.map').write_text('different map\n')
        result = json.loads((self.repeat / 'build-result.json').read_text())
        result['artifacts']['explorer.map'] = preflight.sha((self.repeat / 'explorer.map').read_bytes())
        preflight.write_json(self.repeat / 'build-result.json', result)
        with self.assertRaisesRegex(ValueError, 'не совпала побайтово'):
            package_release.verify_builds(self.first, self.repeat, self.source)

    def test_reject_old_host_build_even_with_passing_test_process(self):
        record = json.loads((self.host / 'build-result.json').read_text())
        record['source_sha256'] = '0' * 64
        preflight.write_json(self.host / 'build-result.json', record)
        with self.assertRaisesRegex(ValueError, 'других исходников'):
            self.package()

    def test_reject_missing_sensor_and_uart_test_groups(self):
        record = json.loads((self.host / 'build-result.json').read_text())
        record['tests'] = ['core', 'runtime', 'servo_bus']
        preflight.write_json(self.host / 'build-result.json', record)
        with self.assertRaisesRegex(ValueError, 'все пять групп'):
            self.package()

    def test_reject_changed_host_executable(self):
        (self.host / 'servo_bus_tests').write_bytes(b'old replacement')
        with self.assertRaisesRegex(ValueError, 'Изменён исполняемый host-тест'):
            self.package()

    def test_reject_vector_outside_application_even_inside_sector(self):
        corrupted = bytearray(self.application)
        struct.pack_into('<I', corrupted, 4 * 15, preflight.BASE + 10001)
        with self.assertRaisesRegex(ValueError, 'Вектор вне приложения'):
            preflight.verify_vectors(corrupted)

    def test_reject_elf_load_in_other_sector(self):
        corrupted = bytearray(make_elf(self.application))
        struct.pack_into('<I', corrupted, 52 + 12, preflight.BASE + preflight.SECTOR_BYTES)
        with self.assertRaisesRegex(ValueError, 'за пределами'):
            preflight.elf_segments(corrupted, self.application)

    def test_reject_elf_bin_disagreement(self):
        corrupted = bytearray(make_elf(self.application))
        corrupted[-1] ^= 1
        with self.assertRaisesRegex(ValueError, 'BIN не совпадает'):
            preflight.elf_segments(corrupted, self.application)

    def test_reject_corrupt_hex_checksum_and_out_of_sector(self):
        lines = package_release.ihex(self.application).splitlines()
        lines[1] = lines[1][:-2] + ('00' if lines[1][-2:] != '00' else '01')
        with self.assertRaisesRegex(ValueError, 'контрольная сумма'):
            preflight.ihex_bytes('\n'.join(lines))
        with self.assertRaisesRegex(ValueError, 'за сектор'):
            preflight.ihex_bytes(package_release.ihex(bytes(preflight.SECTOR_BYTES + 1)))

    def test_reject_current_recovery_mismatch(self):
        self.baseline_path.write_bytes(bytes(preflight.SECTOR_BYTES))
        with self.assertRaisesRegex(ValueError, 'точный установленный'):
            self.package()

    def test_reject_replacing_existing_release(self):
        self.package()
        with self.assertRaisesRegex(ValueError, 'Не перезаписываем'):
            self.package()

    def fake_programmer(self, command, **kwargs):
        self.calls.append(command)
        if command[-2:] == ['-ob', 'displ']:
            return subprocess.CompletedProcess(command, 0, OPTION_TEXT, '')
        self.assertEqual(command[-4], '-u')
        address = int(command[-3], 0)
        size = int(command[-2], 0)
        contents = {0x1ff1e880: struct.pack('<H', self.flash_kib),
                    0x1ff1e800: bytes.fromhex(self.read_uid), preflight.BASE: self.read_sector}
        self.assertEqual(len(contents[address]), size)
        Path(command[-1]).write_bytes(contents[address])
        return subprocess.CompletedProcess(command, 0, 'Upload complete\n', '')

    def prepare_preflight(self):
        self.package()
        self.calls, self.flash_kib, self.read_uid, self.read_sector = [], 1024, self.uid, self.baseline

    def run_preflight(self):
        with mock.patch.object(preflight, 'INSTALLED_SHA', self.baseline_sha), \
             mock.patch.object(preflight.subprocess, 'run', side_effect=self.fake_programmer):
            return preflight.read_only_preflight(self.destination, '/dev/cu.usbserial-fixture',
                                                  self.root / 'session', '/mock/programmer')

    def test_preflight_reads_sector_once_and_never_writes_or_resets(self):
        self.prepare_preflight()
        result = self.run_preflight()
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(sum(preflight.BASE == int(call[-3], 0) for call in self.calls if call[-4] == '-u'), 1)
        self.assertFalse(result['flash_performed'])
        self.assertFalse(result['revision_live_read'])
        self.assertEqual(result['uid'], self.uid)
        self.assertEqual((self.root / 'session/before-sector0.bin').read_bytes(), self.baseline)
        self.assertFalse(any(token in command for command in self.calls
                             for token in ('-d', '-e', '-rst', '-g', '--skipErase')))

    def test_cube_default_flash_text_never_overrides_actual_size(self):
        self.prepare_preflight()
        self.flash_kib = 2048
        with self.assertRaisesRegex(ValueError, 'Реальный размер flash'):
            self.run_preflight()
        self.assertFalse((self.root / 'session/preflight.json').exists())

    def test_preflight_rejects_different_board_uid(self):
        self.prepare_preflight()
        self.read_uid = '0' * 24
        with self.assertRaisesRegex(ValueError, 'UID отличается'):
            self.run_preflight()

    def test_preflight_preserves_mismatched_sector_and_rejects_it(self):
        self.prepare_preflight()
        self.read_sector = bytes(preflight.SECTOR_BYTES)
        with self.assertRaisesRegex(ValueError, 'Текущий сектор отличается'):
            self.run_preflight()
        self.assertEqual((self.root / 'session/before-sector0.bin').read_bytes(), self.read_sector)

    def test_reject_invalid_option_bytes_without_unlock(self):
        with self.assertRaisesRegex(ValueError, 'не соответствуют'):
            preflight.option_values(OPTION_TEXT.replace('RDP: 0xAA', 'RDP: 0xBB'))

    def test_offline_verify_never_invokes_process(self):
        self.package()
        with mock.patch.object(preflight, 'INSTALLED_SHA', self.baseline_sha), \
             mock.patch.object(preflight.subprocess, 'run') as run:
            preflight.verify_package(self.destination)
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
