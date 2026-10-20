from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
s=(WORK/'pin_relocation_implementation_20261020/verify_cached.py').read_text(encoding='utf-8')
s=s.replace('Automatic unsupported-pin relocation','Automatic printed-evidence repair')
s=s.replace('pin_geometry_implementation_20261020/deployed','pin_pairing_implementation_20261020/deployed')
s=s.replace('agent.finalize_pin_relocation(', 'agent.finalize_pin_evidence_repair(')
s=s.replace("previous=WORK/'page2_pin_relocation_20261020/ocr'/(key+'.json')", "previous=WORK/'pin_geometry_implementation_20261020/ocr_cache'/(key+'.json')")
start=s.index("    for owner,pins in baseline['pins'].items():")
end=s.index("    write(out/f'{case}_Pred.json'",start)
s=s[:start]+'''    changed={(g['owner'],'pin_'+g['number']):g for g in audit['changes']}
    assert result['nets']==baseline['nets']
    for owner,pins in baseline['pins'].items():
        for key,pin in pins.items():
            actual=result['pins'].get(owner,{}).get(key)
            assert actual is not None
            change=changed.get((owner,key))
            if change is None:assert actual==pin
            elif change['status']=='restored_observed_full_label':
                assert actual['point']==pin['point'] and actual['pinname']==change['after_name']
            else:
                assert actual['pinname']==pin['pinname'] and actual['point']==change['after']
'''+s[end:]
(ROOT/'verify_cached.py').write_text(s,encoding='utf-8')
s=(WORK/'pin_pairing_implementation_20261020/verify_live_ocr.py').read_text(encoding='utf-8')
s=s.replace("if r['case']=='4'", "if r['case']=='5'")
s=s.replace('pin_relocation_implementation_20261020/deployed/4_Pred.json','pin_pairing_implementation_20261020/deployed/5_Pred.json')
s=s.replace('component_results/4_Pred.json','component_results/5_Pred.json').replace('pcb_open_gt_in/4.png','pcb_open_gt_in/5.png')
s=s.replace('staged/4_Pred.json','staged/5_Pred.json').replace('agent.finalize_pin_pairing(', 'agent.finalize_pin_evidence_repair(')
(ROOT/'verify_live_ocr.py').write_text(s,encoding='utf-8')
s=(WORK/'pin_pairing_implementation_20261020/install.py').read_text(encoding='utf-8')
(ROOT/'install.py').write_text(s.replace('.pcb-pin-pairing-', '.pcb-pin-evidence-'),encoding='utf-8')
print('Prepared validation/install scripts.')
