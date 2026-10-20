from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
s=(WORK/'pin_relocation_implementation_20261020/verify_cached.py').read_text(encoding='utf-8')
s=s.replace('Automatic unsupported-pin relocation','Automatic shared-number pin pairing')
s=s.replace('pin_geometry_implementation_20261020/deployed','pin_relocation_implementation_20261020/deployed')
s=s.replace('agent.finalize_pin_relocation(', 'agent.finalize_pin_pairing(')
s=s.replace("len({p['owner'] for p in audit['proposals']})", "len({p['owner'] for p in audit['groups']})")
s=s.replace("previous=WORK/'page2_pin_relocation_20261020/ocr'/(key+'.json')", "previous=WORK/'page4_pin_pairing_20261020/ocr'/(key+'.json')")
s=s.replace("assert result['nets'].keys()==baseline['nets'].keys()", "assert baseline['nets'].keys() <= result['nets'].keys()")
start=s.index("    for owner,pins in baseline['pins'].items():")
end=s.index("    write(out/f'{case}_Pred.json'",start)
s=s[:start]+'''    rekeys={(g['owner'],'pin_'+str(g['old_number'])):'pin_'+g['number'] for g in audit['changes'] if g['old_number'] is not None}
    for owner,pins in baseline['pins'].items():
        for key,pin in pins.items():
            dest=rekeys.get((owner,key),key)
            assert result['pins'].get(owner,{}).get(dest)==pin, 'Unexpected existing pin change'
'''+s[end:]
s=s.replace("'network_changes':len(audit['network_changes'])", "'network_changes':len(audit['network_changes']),'restored_networks':len(audit['restored_networks'])")
(ROOT/'verify_cached.py').write_text(s,encoding='utf-8')
s=(WORK/'pin_relocation_implementation_20261020/verify_live_ocr.py').read_text(encoding='utf-8')
s=s.replace("if r['case']=='2'", "if r['case']=='4'")
s=s.replace('pin_geometry_implementation_20261020/deployed/2_Pred.json','pin_relocation_implementation_20261020/deployed/4_Pred.json')
s=s.replace('component_results/2_Pred.json','component_results/4_Pred.json').replace('pcb_open_gt_in/2.png','pcb_open_gt_in/4.png')
s=s.replace('staged/2_Pred.json','staged/4_Pred.json').replace('agent.finalize_pin_relocation(', 'agent.finalize_pin_pairing(')
(ROOT/'verify_live_ocr.py').write_text(s,encoding='utf-8')
s=(WORK/'pin_relocation_implementation_20261020/install.py').read_text(encoding='utf-8')
(ROOT/'install.py').write_text(s.replace('.pcb-pin-relocation-', '.pcb-pin-pairing-'),encoding='utf-8')
print('Prepared validation/install scripts.')
