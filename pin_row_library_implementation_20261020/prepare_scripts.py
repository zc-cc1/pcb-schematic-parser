from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent;LIVE=Path('D:/pcb5_submission')
library_path='data/component_library/BoxICLibrary.json'
(ROOT/'runtime'/library_path).parent.mkdir(parents=True,exist_ok=True)
(ROOT/'runtime'/library_path).write_bytes((LIVE/library_path).read_bytes())
(ROOT/'runtime/configs').mkdir(parents=True,exist_ok=True)
cfg={'min_confidence':.98,'min_views':2,'max_components':4,'max_ocr_calls':64,'max_extra_calls_per_component':12,'max_model_distance_px':32,
     'library_path':library_path,'library_sha256':hashlib.sha256((LIVE/library_path).read_bytes()).hexdigest()}
(ROOT/'runtime/configs/pin_row_library_v1.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
s=(WORK/'pin_vertical_repair_implementation_20261020/prepare.py').read_text(encoding='utf-8')
s=s.replace('pin_vertical_repair','pin_row_library').replace('repair_vertical_triplet','reconcile_rows_with_library').replace('PCB_PIN_VERTICAL_REPAIR','PCB_PIN_ROW_LIBRARY')
s=s.replace('_VerticalImage','_RowImage').replace('_vertical_','_row_').replace('Vertical repair','Row library repair')
s=s.replace('Reconcile observed vertical identities and measured endpoints atomically.','Reconcile observed horizontal fields and exact-model library contacts.')
s=s.replace("'restored_networks': []", "'component_changes': [], 'network_withheld': []")
s=s.replace("                    'restored_networks': len(_row_audit['restored_networks']),", "                    'changed_components': len(_row_audit['component_changes']),")
s=s.replace("    config_path = ROOT/'configs/pin_row_library_v1.json'", "    records = json.loads((stage/'export/completed_component_records.json').read_text(encoding='utf-8'))['records']\n    config_path = ROOT/'configs/pin_row_library_v1.json'")
s=s.replace("    engine = None", "    library_path = ROOT/config['library_path']\n    if hashlib.sha256(library_path.read_bytes()).hexdigest() != config['library_sha256']:\n        raise ValueError('Bundled row library fingerprint differs')\n    library = json.loads(library_path.read_text(encoding='utf-8'))\n    engine = None")
s=s.replace('reconcile_rows_with_library(target, pins, graph, image_rgb','reconcile_rows_with_library(target, pins, graph, records, library, image_rgb')
s=s.replace('ocr_models=fingerprints)', "ocr_models=fingerprints, library_sha256=config['library_sha256'])")
(ROOT/'prepare.py').write_text(s,encoding='utf-8')
exec(compile(s[:s.index('files=[]')],str(ROOT/'prepare.py'),'exec'),{'__file__':str(ROOT/'prepare.py')})
s=(WORK/'pin_vertical_repair_implementation_20261020/verify_cached.py').read_text(encoding='utf-8')
s=s.replace('pin_nc_recovery_implementation_20261020/deployed','pin_vertical_repair_implementation_20261020/deployed').replace('agent.finalize_pin_vertical_repair','agent.finalize_pin_row_library')
start=s.index("    assert result['components']==baseline['components']");end=s.index("    write(out/f'{case}_Pred.json'",start)
s=s[:start]+'''    component_changes={r['owner']:r for r in audit['component_changes']}
    for owner,comp in baseline['components'].items():
        expected=copy.deepcopy(comp)
        if owner in component_changes:expected['Name']=component_changes[owner]['after']
        assert result['components'][owner]==expected
    assert result['nets'].keys()==baseline['nets'].keys()
    for nid,net in baseline['nets'].items():assert result['nets'][nid]['edges']==net['edges']
    new_fields={(r['owner'],'pin_'+r['number']):r for r in audit['changes']}
    old_fields={(r['owner'],r['old_key']):r for r in audit['changes'] if r['old_key']}
    for owner,pins in baseline['pins'].items():
        for key,pin in pins.items():
            if (owner,key) in old_fields:
                change=old_fields[owner,key]
                assert change['before']==pin and result['pins'][owner]['pin_'+change['number']]==pin
                if key!='pin_'+change['number']:assert key not in result['pins'][owner]
            else:assert result['pins'][owner][key]==pin
    for owner,pins in result['pins'].items():
        for key,pin in pins.items():
            if key not in baseline['pins'].get(owner,{}):assert new_fields[owner,key]['after']==pin
    for change in audit['network_changes']:
        nid=change['network_id']
        members=lambda n:sorted(v.strip() for v in n['hyperGraph'].strip('()').split(',') if v.strip())
        assert members(baseline['nets'][nid])==change['before'] and members(result['nets'][nid])==change['after']
'''+s[end:]
s=s.replace("'restored_networks':len(audit['restored_networks'])", "'changed_components':len(audit['component_changes'])")
(ROOT/'verify_cached.py').write_text(s,encoding='utf-8')
s=(WORK/'pin_vertical_repair_implementation_20261020/verify_live_ocr.py').read_text(encoding='utf-8')
s=s.replace("r['case']=='10'","r['case']=='11'").replace('pin_nc_recovery_implementation_20261020/deployed/10_Pred.json','pin_vertical_repair_implementation_20261020/deployed/11_Pred.json')
s=s.replace('component_results/10_Pred.json','component_results/11_Pred.json').replace('pcb_open_gt_in/10.png','pcb_open_gt_in/11.png').replace('staged/10_Pred.json','staged/11_Pred.json').replace('agent.finalize_pin_vertical_repair','agent.finalize_pin_row_library')
(ROOT/'verify_live_ocr.py').write_text(s,encoding='utf-8')
s=(WORK/'pin_vertical_repair_implementation_20261020/install.py').read_text(encoding='utf-8')
(ROOT/'install.py').write_text(s.replace('.pcb-pin-vertical-repair-','.pcb-pin-row-library-'),encoding='utf-8')
print('Prepared exact bundled-library wrapper and verification.')
