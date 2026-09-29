"""Exercise destructive boundaries using memory in Python, never a serial port."""
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from contextlib import redirect_stdout

SPEC = importlib.util.spec_from_file_location('installer', Path(__file__).parents[1]/'tools/install.py')
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class MemoryProgrammer:
    def __init__(self, directory, baseline):
        self.directory = directory
        self.bank1 = baseline + b'A'*(524288-len(baseline))
        self.bank2 = b'B'*524288
        registers = bytearray(40)
        for offset, value in ((0, 0xaa10), (12, 0xff), (20, 0xff), (28, 0xff), (36, 0x1ff00800)):
            struct.pack_into('<I', registers, offset, value)
        self.identity = installer.chip_description(struct.pack('<I', 0x20030450),
            struct.pack('<I', 1024), bytes(range(12)), registers)
        self.writes = 0
        self.fail_write = False
        self.change_uid = False

    def identify(self, label):
        identity = dict(self.identity)
        if self.change_uid and label == 'prewrite':
            identity['uid'] = 'changed-device'
        return identity

    def command(self, *args):
        assert args == ('-ob', 'displ')

    def read_bank(self, base, size, label):
        result = (self.bank1 if base == installer.BASE else self.bank2)[:size]
        (self.directory/(label+'.bin')).write_bytes(result)
        return result

    def program_sector(self, image):
        self.writes += 1
        if self.fail_write:
            self.bank1 = b'\xff'*installer.SECTOR + self.bank1[installer.SECTOR:]
            raise OSError('simulated interruption after erase')
        self.bank1 = image + self.bank1[installer.SECTOR:]


class InstallationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'application').mkdir()
        (self.root/'recovery').mkdir()
        (self.root/'session').mkdir()
        self.baseline = bytes(range(256))*512
        app = struct.pack('<II', 0x20020000, 0x08000009) + b'new app'*30
        self.image = app + self.baseline[len(app):]
        files = {'explorer-sector0.bin': self.image, 'application/explorer.bin': app,
                 'recovery/CURRENT-sector0.bin': self.baseline}
        for name, content in files.items():
            (self.root/name).write_bytes(content)
        self.digest = installer.sha(self.image)
        (self.root/'manifest.json').write_text(json.dumps(dict(schema=1, hardware_accepted=False,
            files={n: installer.sha(v) for n, v in files.items()}, image_sha256=self.digest)))
        self.p = MemoryProgrammer(self.root/'session', self.baseline)

    def perform(self, digest=None, bench=True):
        with redirect_stdout(io.StringIO()):
            installer.install(self.root, self.p, digest or self.digest, bench)

    def test_complete_install_restores_exact_preimage_and_keeps_other_flash(self):
        before1, before2 = self.p.bank1, self.p.bank2
        self.perform()
        self.assertEqual(self.p.bank1[:installer.SECTOR], self.image)
        self.assertEqual(self.p.bank1[installer.SECTOR:], before1[installer.SECTOR:])
        self.assertEqual(self.p.bank2, before2)
        (self.root/'restore').mkdir()
        self.p.directory = self.root/'restore'
        with redirect_stdout(io.StringIO()):
            installer.restore(self.p, self.root/'session', installer.sha(self.baseline), True)
        self.assertEqual((self.p.bank1, self.p.bank2), (before1, before2))

    def test_exact_hash_and_bench_are_required_before_any_write(self):
        for digest, bench in (('bad-hash', True), (self.digest, False)):
            with self.assertRaises(ValueError):
                self.perform(digest, bench)
        self.assertEqual(self.p.writes, 0)

    def test_different_current_image_is_saved_but_never_overwritten(self):
        self.p.bank1 = b'x' + self.p.bank1[1:]
        with self.assertRaises(ValueError):
            self.perform()
        self.assertTrue((self.p.directory/'before-bank1.bin').exists())
        self.assertEqual(self.p.writes, 0)

    def test_device_change_between_backup_and_write_aborts(self):
        self.p.change_uid = True
        with self.assertRaises(ValueError):
            self.perform()
        self.assertEqual(self.p.writes, 0)

    def test_interruption_after_erase_leaves_verified_recovery(self):
        self.p.fail_write = True
        with self.assertRaises(OSError):
            self.perform()
        self.assertTrue((self.root/'session/backup.json').exists())
        self.p.fail_write = False
        (self.root/'restore').mkdir()
        self.p.directory = self.root/'restore'
        with redirect_stdout(io.StringIO()):
            installer.restore(self.p, self.root/'session', installer.sha(self.baseline), True)
        self.assertEqual(self.p.bank1[:installer.SECTOR], self.baseline)

    def test_wrong_chip_or_bank_layout_rejected(self):
        registers = bytearray(40)
        for offset, value in ((0, 0xaa10), (12, 0xff), (20, 0xff), (28, 0xff), (36, 0x1ff00800)):
            struct.pack_into('<I', registers, offset, value)
        for code, size in ((0x20030451, 1024), (0x10030450, 1024), (0x20030450, 128), (0x20030450, 0xffff)):
            with self.assertRaises(ValueError):
                installer.chip_description(struct.pack('<I', code), struct.pack('<I', size), bytes(12), registers)
        struct.pack_into('<I', registers, 0, 0x8000aa10)
        with self.assertRaises(ValueError):
            installer.chip_description(struct.pack('<I', 0x20030450), struct.pack('<I', 1024), bytes(12), registers)

    def test_uart_bootloader_option_text_is_validated(self):
        text = '''
Chip ID: 0x450
NVM size  : 2 MBytes (default)
RDP : 0xAA
IWDG1_SW : 0x1
SWAP_BANK : 0x0
BOOT_CM7_ADD0: 0x800
PROT_AREA_START1: 0xFF
PROT_AREA_END1: 0x0
SEC_AREA_START1: 0xFF
SEC_AREA_END1: 0x0
nWRP0 : 0x1
'''
        result = installer.chip_description_from_option_text(text, bytes(range(12)))
        self.assertEqual((result['device'], result['flash_kib']), (0x450, 2048))
        self.assertEqual(result['bank_bytes'], 1024 * 1024)
        for changed in (text.replace('Chip ID: 0x450', 'Chip ID: 0x451'),
                        text.replace('SWAP_BANK : 0x0', 'SWAP_BANK : 0x1'),
                        text.replace('nWRP0 : 0x1', 'nWRP0 : 0x0')):
            with self.assertRaises(ValueError):
                installer.chip_description_from_option_text(changed, bytes(range(12)))

    def test_package_tampering_rejected_before_device(self):
        (self.root/'explorer-sector0.bin').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.perform()
        self.assertEqual(self.p.writes, 0)


if __name__ == '__main__':
    unittest.main()
