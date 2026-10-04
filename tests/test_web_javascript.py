import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest


ROOT=Path(__file__).resolve().parents[1]


class WebJavascriptTests(unittest.TestCase):
    def test_main_inline_script_parses(self):
        node=shutil.which('node')
        if node is None:self.skipTest('node is unavailable')
        html=(ROOT/'src/index.html').read_text()
        start=html.index('<script>')+len('<script>');end=html.index('</script>',start)
        with tempfile.TemporaryDirectory() as folder:
            script=Path(folder)/'index.js';script.write_text(html[start:end])
            result=subprocess.run([node,'--check',str(script)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
