"""Adapt the established cached live-entrypoint verifier to the new hook."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
s=(WORK/'pin_row_library_implementation_20261020/verify_entrypoint.py').read_text(encoding='utf-8')
def replace(a,b):
    global s
    assert a in s,a
    s=s.replace(a,b)
replace('original_row=agent.finalize_pin_row_library','original_row=agent.finalize_pin_row_library\noriginal_numeric=agent.finalize_pin_numeric_connector')
replace("cache_roots=[ROOT/'ocr_cache',","cache_roots=[ROOT/'ocr_cache',WORK/'pin_row_library_implementation_20261020/ocr_cache',")
replace("for case in ['1','2','4','5','6','8','10','11']:","for case in ['1','2','4','5','6','8','10','11','12','13']:")
replace("baseline=read(WORK/f'pin_vertical_repair_implementation_20261020/deployed/", "baseline=read(WORK/f'pin_row_library_implementation_20261020/deployed/")
replace("        else:assert result['components']==before['components']", """        elif function is original_numeric:
            for owner,comp in before['components'].items():assert result['components'][owner]==comp
            assert set(result['components'])-set(before['components'])=={r['owner'] for r in audit['component_changes']}
        else:assert result['components']==before['components']""")
replace('        calls.append(audit);return result,audit','        return result,audit\n    def numeric_hook(*args):\n        result,audit=checked_call(original_numeric,*args)\n        calls.append(audit);return result,audit')
replace("patch.object(agent,'finalize_pin_row_library',side_effect=row_hook):", "patch.object(agent,'finalize_pin_row_library',side_effect=row_hook),patch.object(agent,'finalize_pin_numeric_connector',side_effect=numeric_hook):")
replace("row['pin_row_library']", "row['pin_numeric_connector']")
replace("logs/{case}_pin_row_library.json", "logs/{case}_pin_numeric_connector.json")
replace("'row_library_log_written'", "'numeric_connector_log_written'")
replace("    verified=read(ROOT/f'staged/{case}_audit.json')", """    verified=read(ROOT/f'staged/{case}_audit.json')
    for change in verified['component_changes']:
        owner=change['owner']
        assert actual['components'][owner]==expected['components'][owner]
        assert actual['pins'][owner]==expected['pins'][owner]
    for nid,net in expected['nets'].items():
        assert actual['nets'][nid]==net, ('Final network differs',case,nid)""")
(ROOT/'verify_entrypoint.py').write_text(s,encoding='utf-8')
print('Prepared 10 same-image cached export checks including page 13.')
