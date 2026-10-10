import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from learning_environment import configure,config_notes


class LearningEnvironmentTests(unittest.TestCase):
    def test_inherited_unwritable_cache_is_overridden_under_project_data(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{
            "HF_HOME":"/inaccessible/cache","HF_DATASETS_CACHE":"/inaccessible/datasets",
            "HF_HUB_CACHE":"/inaccessible/hub","TORCH_HOME":"/inaccessible/torch"}):
            root=Path(directory).resolve()
            configured=configure(root)
            for name,path in configured.items():
                self.assertTrue(Path(path).is_relative_to(root/"data"))
                self.assertTrue(Path(path).is_dir())
                self.assertEqual(os.environ[name],path)
                probe=Path(path)/"write-test";probe.write_text("readable")
                self.assertEqual(probe.read_text(),"readable")
            self.assertEqual(json.loads(config_notes(configured))["explorer_cache_environment"],configured)

    def test_mobile_selftest_configures_cache_before_framework_import(self):
        source=Path(__file__).resolve().parents[1]/"bin/selftest-mobile-learning.py"
        function=next(node for node in ast.parse(source.read_text()).body if isinstance(node,ast.FunctionDef) and node.name=="main")
        configured=next(index for index,node in enumerate(function.body) if isinstance(node,ast.Assign) and
            isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=="configure")
        for index,node in enumerate(function.body):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                modules=[item.name for item in node.names] if isinstance(node,ast.Import) else [node.module or ""]
                if any(name.startswith(("torch","datasets","lerobot")) for name in modules):
                    self.assertLess(configured,index)


if __name__=="__main__":unittest.main()
