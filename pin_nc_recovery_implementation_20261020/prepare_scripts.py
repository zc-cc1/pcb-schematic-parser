from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
(ROOT/'runtime/configs').mkdir(parents=True,exist_ok=True)
(ROOT/'runtime/configs/pin_nc_recovery_v1.json').write_text(json.dumps({'min_confidence':.98,'min_views':2,'max_components':3,'max_proposals':8,'max_ocr_calls':48,'min_row_spacing_px':18},indent=2)+'\n',encoding='utf-8')
source=(WORK/'pin_number_cycle_implementation_20261020/prepare.py').read_text(encoding='utf-8')
source=source.replace('pin_number_cycle','pin_nc_recovery').replace('reconcile_bottom_number_cycles','recover_crossed_nc').replace('PCB_PIN_NUMBER_CYCLE','PCB_PIN_NC_RECOVERY')
source=source.replace('_CycleImage','_NCImage').replace('_cycle_','_nc_')
source=source.replace('Corroborate and atomically permute bottom-row pin identities.','Recover only directly corroborated crossed NC lead identities.')
source=source.replace('Pin number cycle','NC recovery')
source=source.replace("    records = json.loads((stage/'export/completed_component_records.json').read_text(encoding='utf-8'))['records']\n",'')
source=source.replace('recover_crossed_nc(target, pins, graph, records, image_rgb','recover_crossed_nc(target, pins, image_rgb')
(ROOT/'prepare.py').write_text(source,encoding='utf-8')
source=(WORK/'pin_number_cycle_implementation_20261020/verify_cached.py').read_text(encoding='utf-8')
source=source.replace('pin_evidence_implementation_20261020/deployed','pin_number_cycle_implementation_20261020/deployed').replace('agent.finalize_pin_number_cycle','agent.finalize_pin_nc_recovery')
start=source.index('    rekeys=');end=source.index("    write(out/f'{case}_Pred.json'",start)
source=source[:start]+'''    assert result['nets']==baseline['nets']
    expected_added={(v['owner'],'pin_'+v['number']) for v in audit['changes']}
    actual_added=set()
    for owner,pins in result['pins'].items():
        for key,pin in pins.items():
            if key in baseline['pins'].get(owner,{}):assert pin==baseline['pins'][owner][key]
            else:actual_added.add((owner,key))
    for owner,pins in baseline['pins'].items():
        assert all(result['pins'][owner][key]==pin for key,pin in pins.items())
    assert actual_added==expected_added
'''+source[end:]
(ROOT/'verify_cached.py').write_text(source,encoding='utf-8')
source=(WORK/'pin_number_cycle_implementation_20261020/verify_live_ocr.py').read_text(encoding='utf-8')
source=source.replace("r['case']=='6'","r['case']=='8'").replace('pin_evidence_implementation_20261020/deployed/6_Pred.json','pin_number_cycle_implementation_20261020/deployed/8_Pred.json')
source=source.replace('component_results/6_Pred.json','component_results/8_Pred.json').replace('pcb_open_gt_in/6.png','pcb_open_gt_in/8.png').replace('staged/6_Pred.json','staged/8_Pred.json').replace('agent.finalize_pin_number_cycle','agent.finalize_pin_nc_recovery')
(ROOT/'verify_live_ocr.py').write_text(source,encoding='utf-8')
source=(WORK/'pin_number_cycle_implementation_20261020/install.py').read_text(encoding='utf-8')
(ROOT/'install.py').write_text(source.replace('.pcb-pin-number-cycle-','.pcb-pin-nc-recovery-'),encoding='utf-8')
# Stage the agent wrapper now, but only prepare the guarded release after tests/docs exist.
prepare=(ROOT/'prepare.py').read_text(encoding='utf-8');prefix=prepare[:prepare.index('files=[]')]
exec(compile(prefix,str(ROOT/'prepare.py'),'exec'),{'__file__':str(ROOT/'prepare.py')})
print('Prepared runtime wrapper and retained-model verification.')
