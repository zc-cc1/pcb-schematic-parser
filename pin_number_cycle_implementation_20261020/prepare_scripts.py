from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
s=(WORK/'pin_evidence_implementation_20261020/prepare.py').read_text(encoding='utf-8')
s=s.replace('pin_evidence_repair','pin_number_cycle').replace('repair_observed_pin_evidence','reconcile_bottom_number_cycles')
s=s.replace('PCB_PIN_EVIDENCE_REPAIR','PCB_PIN_NUMBER_CYCLE').replace('_EvidenceImage','_CycleImage').replace('_evidence_','_cycle_')
s=s.replace('Restore corroborated printed labels and isolated collision points.', 'Corroborate and atomically permute bottom-row pin identities.')
s=s.replace('Pin evidence','Pin number cycle')
(ROOT/'prepare.py').write_text(s,encoding='utf-8')
s=(WORK/'pin_relocation_implementation_20261020/verify_cached.py').read_text(encoding='utf-8')
s=s.replace('Automatic unsupported-pin relocation','Automatic closed number permutation')
s=s.replace('pin_geometry_implementation_20261020/deployed','pin_evidence_implementation_20261020/deployed')
s=s.replace('agent.finalize_pin_relocation(', 'agent.finalize_pin_number_cycle(')
s=s.replace("len({p['owner'] for p in audit['proposals']})", "len({p['owner'] for p in audit['groups']})")
s=s.replace("previous=WORK/'page2_pin_relocation_20261020/ocr'/(key+'.json')", "previous=WORK/'pin_geometry_implementation_20261020/ocr_cache'/(key+'.json')")
start=s.index("    for owner,pins in baseline['pins'].items():")
end=s.index("    write(out/f'{case}_Pred.json'",start)
s=s[:start]+'''    rekeys={(g['owner'],'pin_'+g['old_number']):'pin_'+g['number'] for g in audit['changes']}
    for owner,pins in baseline['pins'].items():
        assert pins.keys()==result['pins'].get(owner,{}).keys()
        for key,pin in pins.items():
            assert result['pins'][owner][rekeys.get((owner,key),key)]==pin
    for nid,net in baseline['nets'].items():
        before=sorted(v.strip() for v in net['hyperGraph'].strip('()').split(',') if v.strip())
        after=sorted(v.strip() for v in result['nets'][nid]['hyperGraph'].strip('()').split(',') if v.strip())
        assert len(before)==len(after)
'''+s[end:]
(ROOT/'verify_cached.py').write_text(s,encoding='utf-8')
s=(WORK/'pin_evidence_implementation_20261020/verify_live_ocr.py').read_text(encoding='utf-8')
s=s.replace("if r['case']=='5'", "if r['case']=='6'")
s=s.replace('pin_pairing_implementation_20261020/deployed/5_Pred.json','pin_evidence_implementation_20261020/deployed/6_Pred.json')
s=s.replace('component_results/5_Pred.json','component_results/6_Pred.json').replace('pcb_open_gt_in/5.png','pcb_open_gt_in/6.png')
s=s.replace('staged/5_Pred.json','staged/6_Pred.json').replace('agent.finalize_pin_evidence_repair(', 'agent.finalize_pin_number_cycle(')
(ROOT/'verify_live_ocr.py').write_text(s,encoding='utf-8')
s=(WORK/'pin_evidence_implementation_20261020/install.py').read_text(encoding='utf-8')
(ROOT/'install.py').write_text(s.replace('.pcb-pin-evidence-', '.pcb-pin-number-cycle-'),encoding='utf-8')
print('Prepared staging, verification and installation scripts.')
