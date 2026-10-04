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

    def test_manual_view_and_both_inputs_share_explicit_resume(self):
        html=(ROOT/'src/index.html').read_text()
        self.assertIn("showTab('manual');loadControlScheme();updateControlHelp();selectInput('keyboard')",html)
        self.assertIn("api('teleop/resume',{source:input,observing:true})",html)
        self.assertIn("e.code==='Digit1'",html)
        self.assertIn("frame?raw=true",html)
        self.assertIn('id="loginDialog"',html)
        self.assertNotIn("prompt('Ключ",html)

    def test_sock_training_has_a_single_visible_quick_start(self):
        html=(ROOT/'src/index.html').read_text()
        self.assertIn('Подготовить: носок → корзина',html)
        self.assertIn('НАЧАТЬ СЕРИЮ И ПЕРВЫЙ ПОКАЗ',html)
        self.assertIn('function prepareSockTraining()',html)
        self.assertIn('async function enableTrainingManual()',html)
        self.assertIn("$('workflowSkill').value='mobile_pick_place'",html)
        self.assertIn('id="recordingBadge"',html)
        self.assertIn('id="trainingDetections"',html)
        self.assertIn('function renderTrainingStatus(',html)
        self.assertIn("api('frame')",html)


if __name__=='__main__':unittest.main()
