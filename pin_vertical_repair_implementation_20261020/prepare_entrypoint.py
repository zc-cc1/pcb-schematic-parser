"""Replay the actual complete export path with all existing repair hooks."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
for config in Path('D:/pcb5_submission/configs').glob('*.json'):
    if config.name!='pin_vertical_repair_v1.json':(ROOT/'runtime/configs'/config.name).write_bytes(config.read_bytes())
s=(WORK/'pin_nc_recovery_implementation_20261020/verify_entrypoint.py').read_text(encoding='utf-8')
s=s.replace('original_nc=agent.finalize_pin_nc_recovery','original_nc=agent.finalize_pin_nc_recovery\noriginal_vertical=agent.finalize_pin_vertical_repair')
s=s.replace("cache_roots=[ROOT/'ocr_cache',", "cache_roots=[ROOT/'ocr_cache',WORK/'pin_nc_recovery_implementation_20261020/ocr_cache',")
s=s.replace("for case in ['1','2','4','5','6','8']:","for case in ['1','2','4','5','6','8','10']:")
s=s.replace("baseline=read(WORK/f'pin_number_cycle_implementation_20261020/deployed/", "baseline=read(WORK/f'pin_nc_recovery_implementation_20261020/deployed/")
s=s.replace('        calls.append(audit);return result,audit', '        return result,audit\n    def vertical_hook(*args):\n        result,audit=checked_call(original_vertical,*args)\n        calls.append(audit);return result,audit')
s=s.replace("patch.object(agent,'finalize_pin_nc_recovery',side_effect=nc_hook):", "patch.object(agent,'finalize_pin_nc_recovery',side_effect=nc_hook),patch.object(agent,'finalize_pin_vertical_repair',side_effect=vertical_hook):")
s=s.replace("row['pin_nc_recovery']", "row['pin_vertical_repair']")
s=s.replace("logs/{case}_pin_nc_recovery.json", "logs/{case}_pin_vertical_repair.json")
s=s.replace("'nc_recovery_log_written'", "'vertical_repair_log_written'")
s=s.replace("'ocr_calls':verified['ocr_calls']}", "'ocr_calls':verified['ocr_calls'],'restored_networks':len(verified['restored_networks'])}")
s=s.replace("for restored in prior_pairing['restored_networks']:", "for restored in prior_pairing['restored_networks']+verified['restored_networks']:")
(ROOT/'verify_entrypoint.py').write_text(s,encoding='utf-8')
print('Prepared actual export checks for pages 1, 2, 4, 5, 6, 8, 10')
