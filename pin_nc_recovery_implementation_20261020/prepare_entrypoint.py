"""Verify every existing repair plus NC recovery in the actual export path."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
# The staged agent resolves all existing configurations relative to itself.
# Copy those read-only support files for the smoke test; release_plan still
# deploys only the five explicitly reviewed files.
for config in Path('D:/pcb5_submission/configs').glob('*.json'):
    destination=ROOT/'runtime/configs'/config.name
    if config.name!='pin_nc_recovery_v1.json':destination.write_bytes(config.read_bytes())
s=(WORK/'pin_number_cycle_implementation_20261020/verify_entrypoint.py').read_text(encoding='utf-8')
s=s.replace('import copy','import copy\nimport argparse')
s=s.replace("sys.path.insert(0,str(LIVE/'src'))", "sys.path.insert(0,str(LIVE/'src'))\nparser=argparse.ArgumentParser();parser.add_argument('--deployed',action='store_true');args=parser.parse_args()\nif not args.deployed:sys.path.insert(0,str(ROOT/'runtime'))\nimport pcb_parser\nif not args.deployed:pcb_parser.__path__.insert(0,str(ROOT/'runtime/src/pcb_parser'))")
s=s.replace('original_cycle=agent.finalize_pin_number_cycle','original_cycle=agent.finalize_pin_number_cycle\noriginal_nc=agent.finalize_pin_nc_recovery')
s=s.replace("cache_roots=[ROOT/'ocr_cache',", "cache_roots=[ROOT/'ocr_cache',WORK/'pin_number_cycle_implementation_20261020/ocr_cache',")
s=s.replace("for case in ['1','2','4','5','6']:", "for case in ['1','2','4','5','6','8']:")
s=s.replace("baseline=read(WORK/f'pin_evidence_implementation_20261020/deployed/", "baseline=read(WORK/f'pin_number_cycle_implementation_20261020/deployed/")
s=s.replace('        calls.append(audit);return result,audit', '        return result,audit\n    def nc_hook(*args):\n        before=copy.deepcopy(args[0])\n        result,audit=checked_call(original_nc,*args)\n        assert result[\'nets\']==before[\'nets\']\n        calls.append(audit);return result,audit')
s=s.replace("patch.object(agent,'finalize_pin_number_cycle',side_effect=cycle_hook):", "patch.object(agent,'finalize_pin_number_cycle',side_effect=cycle_hook),patch.object(agent,'finalize_pin_nc_recovery',side_effect=nc_hook):")
s=s.replace("row['pin_number_cycle']", "row['pin_nc_recovery']")
s=s.replace("logs/{case}_pin_number_cycle.json", "logs/{case}_pin_nc_recovery.json")
s=s.replace("'number_cycle_log_written'", "'nc_recovery_log_written'")
s=s.replace("for change in prior_pairing['network_changes']+verified['network_changes']:", "prior_cycle=read(WORK/f'pin_number_cycle_implementation_20261020/deployed/{case}_audit.json')\n    for change in prior_pairing['network_changes']+prior_cycle['network_changes']+verified['network_changes']:")
s=s.replace("'kept_work':True}", "'kept_work':True,'runtime':'deployed' if args.deployed else 'staged'}")
s=s.replace("ROOT/'entrypoint_check.json'", "ROOT/('entrypoint_deployed_check.json' if args.deployed else 'entrypoint_check.json')")
(ROOT/'verify_entrypoint.py').write_text(s,encoding='utf-8')
print('Prepared actual export check for pages 1, 2, 4, 5, 6, 8')
