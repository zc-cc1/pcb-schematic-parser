from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
s=(WORK/'pin_relocation_implementation_20261020/verify_entrypoint.py').read_text(encoding='utf-8')
s=s.replace('original_relocation=agent.finalize_pin_relocation',
            'original_relocation=agent.finalize_pin_relocation\noriginal_pairing=agent.finalize_pin_pairing')
s=s.replace("cache_roots=[ROOT/'ocr_cache'", "cache_roots=[ROOT/'ocr_cache',WORK/'pin_relocation_implementation_20261020/ocr_cache'")
s=s.replace("for case in ['1','2']:","for case in ['1','2','4']:")
s=s.replace('pin_geometry_implementation_20261020/deployed','pin_relocation_implementation_20261020/deployed')
s=s.replace("    def relocation_hook(*args):\n        result,audit=checked_call(original_relocation,*args);calls.append(audit);return result,audit", 
'''    def relocation_hook(*args):return checked_call(original_relocation,*args)
    def pairing_hook(*args):
        result,audit=checked_call(original_pairing,*args);calls.append(audit);return result,audit''')
s=s.replace("side_effect=relocation_hook):", "side_effect=relocation_hook),patch.object(agent,'finalize_pin_pairing',side_effect=pairing_hook):")
s=s.replace("'ocr_calls':verified['ocr_calls']", "'restored_networks':len(verified['restored_networks']),'ocr_calls':verified['ocr_calls']")
s=s.replace("row['pin_relocation']", "row['pin_pairing']")
s=s.replace("logs/{case}_pin_relocation.json", "logs/{case}_pin_pairing.json")
s=s.replace("'relocation_log_written'", "'pairing_log_written'")
anchor="    results.append({'case':case"
checks='''    for restored in verified['restored_networks']:
        nid=restored['network_id']
        assert actual['nets'][nid]==expected['nets'][nid], 'Restored branch differs from verified output'
    for change in verified['network_changes']:
        if 'after' in change:
            nid=change['network_id']
            members=sorted(v.strip() for v in actual['nets'][nid]['hyperGraph'].strip('()').split(',') if v.strip())
            assert members==change['after'], 'Rekeyed membership differs from verified output'
'''
assert anchor in s;s=s.replace(anchor,checks+anchor)
(ROOT/'verify_entrypoint.py').write_text(s,encoding='utf-8')
print('Prepared three-page live-entrypoint smoke check.')
