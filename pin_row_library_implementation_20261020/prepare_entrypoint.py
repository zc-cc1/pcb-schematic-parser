"""Replay all deployed repair hooks with same-image model and OCR caches."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent;LIVE=Path('D:/pcb5_submission')
for config in (LIVE/'configs').glob('*.json'):
    if config.name!='pin_row_library_v1.json':(ROOT/'runtime/configs'/config.name).write_bytes(config.read_bytes())
library='data/component_library/BoxICLibrary.json'
(ROOT/'runtime'/library).write_bytes((LIVE/library).read_bytes())
s=(WORK/'pin_vertical_repair_implementation_20261020/verify_entrypoint.py').read_text(encoding='utf-8')
s=s.replace('original_vertical=agent.finalize_pin_vertical_repair','original_vertical=agent.finalize_pin_vertical_repair\noriginal_row=agent.finalize_pin_row_library')
s=s.replace("cache_roots=[ROOT/'ocr_cache',", "cache_roots=[ROOT/'ocr_cache',WORK/'pin_vertical_repair_implementation_20261020/ocr_cache',")
s=s.replace("for case in ['1','2','4','5','6','8','10']:","for case in ['1','2','4','5','6','8','10','11']:")
s=s.replace("baseline=read(WORK/f'pin_nc_recovery_implementation_20261020/deployed/", "baseline=read(WORK/f'pin_vertical_repair_implementation_20261020/deployed/")
s=s.replace("        assert target==before and result['components']==before['components']", """        assert target==before
        if function is original_row:
            changed={r['owner']:r for r in audit['component_changes']}
            for owner,comp in before['components'].items():
                expected_comp=copy.deepcopy(comp)
                if owner in changed:expected_comp['Name']=changed[owner]['after']
                assert result['components'][owner]==expected_comp
        else:assert result['components']==before['components']""")
s=s.replace('        calls.append(audit);return result,audit', '        return result,audit\n    def row_hook(*args):\n        result,audit=checked_call(original_row,*args)\n        calls.append(audit);return result,audit')
s=s.replace("patch.object(agent,'finalize_pin_vertical_repair',side_effect=vertical_hook):", "patch.object(agent,'finalize_pin_vertical_repair',side_effect=vertical_hook),patch.object(agent,'finalize_pin_row_library',side_effect=row_hook):")
s=s.replace("row['pin_vertical_repair']", "row['pin_row_library']").replace("logs/{case}_pin_vertical_repair.json", "logs/{case}_pin_row_library.json").replace("'vertical_repair_log_written'", "'row_library_log_written'")
s=s.replace("'restored_networks':len(verified['restored_networks'])", "'changed_components':len(verified['component_changes'])")
s=s.replace("prior_pairing['restored_networks']+verified['restored_networks']", "prior_pairing['restored_networks']+verified.get('restored_networks',[])")
s=s.replace("if comp['type'] in {'box','ic'}:assert actual['pins'].get(owner)==expected['pins'].get(owner),owner", "if comp['type'] in {'box','ic'}:\n            assert actual['pins'].get(owner)==expected['pins'].get(owner),owner\n            assert actual['components'][owner]==comp,owner")
(ROOT/'verify_entrypoint.py').write_text(s,encoding='utf-8')
print('Prepared actual export checks including page 11 and exact library loading.')
