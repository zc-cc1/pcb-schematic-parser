"""Prepare exact-current staging and reusable local validation scripts."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
(ROOT/'runtime/configs').mkdir(parents=True,exist_ok=True)
config={'min_confidence':.98,'min_views':2,'max_components':2,'max_ocr_calls':48,'min_lead_px':24,'max_lead_px':60,'max_length_spread_px':4}
(ROOT/'runtime/configs/pin_vertical_repair_v1.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
s=(WORK/'pin_nc_recovery_implementation_20261020/prepare.py').read_text(encoding='utf-8')
s=s.replace('pin_nc_recovery','pin_vertical_repair').replace('recover_crossed_nc','repair_vertical_triplet').replace('PCB_PIN_NC_RECOVERY','PCB_PIN_VERTICAL_REPAIR')
s=s.replace('NC recovery','Vertical repair').replace('Recover only directly corroborated crossed NC lead identities.','Reconcile observed vertical identities and measured endpoints atomically.')
s=s.replace('repair_vertical_triplet(target, pins, image_rgb','repair_vertical_triplet(target, pins, graph, image_rgb')
s=s.replace('_NCImage','_VerticalImage').replace('_nc_','_vertical_')
s=s.replace("'ocr_calls': 0}","'ocr_calls': 0, 'restored_networks': []}")
s=s.replace("'changed_pins': len(_vertical_audit['changes']),", "'changed_pins': len(_vertical_audit['changes']),\n                    'restored_networks': len(_vertical_audit['restored_networks']),")
(ROOT/'prepare.py').write_text(s,encoding='utf-8')
prefix=s[:s.index('files=[]')];exec(compile(prefix,str(ROOT/'prepare.py'),'exec'),{'__file__':str(ROOT/'prepare.py')})
s=(WORK/'pin_nc_recovery_implementation_20261020/verify_cached.py').read_text(encoding='utf-8')
s=s.replace('pin_number_cycle_implementation_20261020/deployed','pin_nc_recovery_implementation_20261020/deployed').replace('agent.finalize_pin_nc_recovery','agent.finalize_pin_vertical_repair')
start=s.index("    assert result['components']==baseline['components']");end=s.index("    write(out/f'{case}_Pred.json'",start)
s=s[:start]+'''    assert result['components']==baseline['components']
    assert baseline['nets'].keys()<=result['nets'].keys()
    for nid,net in baseline['nets'].items():assert net['edges']==result['nets'][nid]['edges']
    changes={(r['owner'],'pin_'+r['number']):r for r in audit['changes']}
    for owner,pins in baseline['pins'].items():
        for key,pin in pins.items():
            if (owner,key) in changes:
                assert pin==changes[owner,key]['before'] and result['pins'][owner][key]==changes[owner,key]['after']
            else:assert result['pins'][owner][key]==pin
    for change in audit['network_changes']:
        nid=change['network_id']
        def members(net):return sorted(v.strip() for v in net['hyperGraph'].strip('()').split(',') if v.strip())
        assert members(baseline['nets'][nid])==change['before']
        assert members(result['nets'][nid])==change['after']
    assert set(result['nets'])-set(baseline['nets'])=={r['network_id'] for r in audit['restored_networks']}
'''+s[end:]
s=s.replace("'network_changes':len(audit['network_changes']),", "'network_changes':len(audit['network_changes']),'restored_networks':len(audit['restored_networks']),")
(ROOT/'verify_cached.py').write_text(s,encoding='utf-8')
s=(WORK/'pin_nc_recovery_implementation_20261020/verify_live_ocr.py').read_text(encoding='utf-8')
s=s.replace("r['case']=='8'","r['case']=='10'").replace('pin_number_cycle_implementation_20261020/deployed/8_Pred.json','pin_nc_recovery_implementation_20261020/deployed/10_Pred.json')
s=s.replace('component_results/8_Pred.json','component_results/10_Pred.json').replace('pcb_open_gt_in/8.png','pcb_open_gt_in/10.png').replace('staged/8_Pred.json','staged/10_Pred.json').replace('agent.finalize_pin_nc_recovery','agent.finalize_pin_vertical_repair')
(ROOT/'verify_live_ocr.py').write_text(s,encoding='utf-8')
s=(WORK/'pin_nc_recovery_implementation_20261020/install.py').read_text(encoding='utf-8')
(ROOT/'install.py').write_text(s.replace('.pcb-pin-nc-recovery-','.pcb-pin-vertical-repair-'),encoding='utf-8')
print('Prepared vertical triplet wrapper and verification scripts.')
