import importlib.util
import argparse
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parent;LIVE=Path('D:/pcb5_submission')
parser=argparse.ArgumentParser();parser.add_argument('--deployed',action='store_true');args=parser.parse_args()
sys.dont_write_bytecode=True
sys.path.insert(0,str(LIVE));sys.path.insert(0,str(LIVE/'src'))
if not args.deployed:sys.path.insert(0,str(ROOT/'runtime'))
import pcb_parser
if not args.deployed:pcb_parser.__path__.insert(0,str(ROOT/'runtime/src/pcb_parser'))
import agent
suite=unittest.TestSuite();loader=unittest.TestLoader()
paths=sorted((LIVE/'tests').glob('test_*.py'))
if not args.deployed:paths=[p for p in paths if p.name!='test_pin_numeric_connector.py']+sorted((ROOT/'runtime/tests').glob('test_*.py'))
for path in paths:
    spec=importlib.util.spec_from_file_location('verify_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    suite.addTests(loader.loadTestsFromModule(module))
result=unittest.TextTestRunner(verbosity=1).run(suite)
(ROOT/('tests_deployed.json' if args.deployed else 'tests_staged.json')).write_text(json.dumps({'runtime':'deployed' if args.deployed else 'staged','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'success':result.wasSuccessful()},indent=2)+'\n',encoding='utf-8')
sys.exit(not result.wasSuccessful())
