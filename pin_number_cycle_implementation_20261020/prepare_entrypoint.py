"""Extend the prior retained-model export check to include number cycles."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = (ROOT.parent / 'pin_evidence_implementation_20261020/verify_entrypoint.py').read_text(encoding='utf-8')
source = source.replace('original_evidence=agent.finalize_pin_evidence_repair', 'original_evidence=agent.finalize_pin_evidence_repair\noriginal_cycle=agent.finalize_pin_number_cycle')
source = source.replace("cache_roots=[ROOT/'ocr_cache',", "cache_roots=[ROOT/'ocr_cache',WORK/'pin_evidence_implementation_20261020/ocr_cache',")
source = source.replace("for case in ['1','2','4','5']:", "for case in ['1','2','4','5','6']:")
source = source.replace("baseline=read(WORK/f'pin_pairing_implementation_20261020/deployed/", "baseline=read(WORK/f'pin_evidence_implementation_20261020/deployed/")
source = source.replace('        calls.append(audit);return result,audit', '        return result,audit\n    def cycle_hook(*args):\n        result,audit=checked_call(original_cycle,*args)\n        calls.append(audit);return result,audit')
source = source.replace("patch.object(agent,'finalize_pin_evidence_repair',side_effect=evidence_hook):", "patch.object(agent,'finalize_pin_evidence_repair',side_effect=evidence_hook),patch.object(agent,'finalize_pin_number_cycle',side_effect=cycle_hook):")
source = source.replace("row['pin_evidence_repair']", "row['pin_number_cycle']")
source = source.replace("logs/{case}_pin_evidence_repair.json", "logs/{case}_pin_number_cycle.json")
source = source.replace("'evidence_log_written'", "'number_cycle_log_written'")
(ROOT / 'verify_entrypoint.py').write_text(source,encoding='utf-8')
print('Prepared live export verification for pages 1, 2, 4, 5, 6')
