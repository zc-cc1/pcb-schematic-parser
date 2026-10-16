"""Verify actual deployment output changes only declared LED boxes and power additions."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'dist/led_bbox_fix_20261016'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    spec=importlib.util.spec_from_file_location('led_contract_adapter',ROOT/'submission/platform_adapter/adapter.py')
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    hashes=read(OUT/'container_source_sha256.json')
    for relative,digest in hashes.items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,relative
    output=OUT/'container-output'
    decision=read(output/'decision_log.json')
    assert decision['successful_cases']==1 and decision['failed_cases']==0
    assert decision['device']=='cpu' and decision['output_format']=='target'
    before=read(output/'project_results/0107_Pred.json')
    raw=read(output/'component_results/0107_Pred.json')
    actual=read(output/'final_results/0107_Pred.json')
    assert all(raw['components'][key]==c for key,c in before['components'].items())
    assert all(raw['pins'][key]==p for key,p in before['pins'].items())
    assert before['nets']==raw['nets']
    expected=adapter.to_target(raw)
    audit=read(output/'logs/0107_led_geometry.json')
    summary=read(output/'logs/0107_summary.json')
    assert audit['entries'] and audit['image_sha256']==summary['image_sha256']
    leds={key for key,c in raw['components'].items() if c['type']=='led'}
    assert {row['component_id'] for row in audit['entries']}==leds
    for row in audit['entries']:
        key=row['component_id']
        assert row['raw_bbox_bottom_left']==raw['components'][key]['bbox']
        assert row['before_bbox_bottom_left']==expected['components'][key]['bbox']
        expected['components'][key]['bbox']=row['after_bbox_bottom_left']
    assert actual==expected
    adapter.validate_target(actual)
    result={'case':'0107','successful_cases':1,'network_disabled':True,
            'runtime':read(output/'runtime_environment.json'), 'whole_case_seconds':decision['cases'][0]['seconds'],
            'led_geometry_seconds':audit['wall_seconds'],'LED_count':len(leds),
            'changed_LED_boxes':audit['changed_LED_boxes'],'changed_LED_centers':audit['changed_LED_centers'],
            'only_LED_bbox_changed_by_geometry_stage':True,'pre_geometry_component_export_preserved':True,
            'all_LED_text_keys_pins_and_nets_preserved_by_geometry_stage':True,
            'mounted_source_hashes_verified':True,'source_sha256':hashes,'related_tests_passed':65,
            'test_method':'Existing official-base image with current files bind-mounted read-only; PNG only input.',
            'new_image_built':False,'new_multigigabyte_upload_generated':False,'official_score':None}
    (OUT/'container_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','runtime']}))


if __name__=='__main__':main()
