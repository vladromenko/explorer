"""Derive HAL configuration and the unchanged vendor clock function at build time."""
from pathlib import Path
import re
import sys
import hashlib

root = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
text = (root / 'vendor/board/stm32h7xx_hal_conf.h').read_text()
for name in ('ADC', 'SPI', 'UART', 'IWDG', 'RNG'):
    text = re.sub(r'/\*\s*#define HAL_' + name + r'_MODULE_ENABLED\s*\*/',
                  '#define HAL_' + name + '_MODULE_ENABLED', text)
(out / 'stm32h7xx_hal_conf.h').write_text(text)
source = (root / 'vendor/board/main.c').read_text()
start = source.index('void SystemClock_Config(void)\n{')
end = source.index('\n/* USER CODE BEGIN 4 */', start)
(out / 'clock.c').write_text('/* Clock function from Yahboom Motor example; see sources.json. */\n'
                            '#include "main.h"\n' + source[start:end])
digest = hashlib.sha256()
for folder in ('core', 'src', 'include', 'vendor', 'tools', 'cmake', 'tests'):
    for path in sorted((root / folder).rglob('*')):
        if path.is_file() and path.suffix in ('.c', '.h', '.s', '.py', '.cmake'):
            digest.update(path.relative_to(root).as_posix().encode() + b'\0' + path.read_bytes())
for name in ('CMakeLists.txt', 'target.ld', 'sources.json'):
    digest.update(name.encode() + b'\0' + (root / name).read_bytes())
(out / 'build_identity.h').write_text('#define EX_SOURCE_SHA256 "' + digest.hexdigest() + '"\n'
                                    'static const unsigned char ex_source_digest[32]={' +
                                    ','.join(str(b) for b in digest.digest()) + '};\n')
(out / 'source.sha256').write_text(digest.hexdigest() + '\n')
