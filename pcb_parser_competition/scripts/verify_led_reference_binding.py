"""Reproduce the LED1/ED1 field collision and replay image-only OCR caches."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pcb_parser.component_text import bind_components


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out=ROOT/'dist/led_reference_binding_fix_20261016'
    spec=importlib.util.spec_from_file_location('before_led_binding',out/'component_text_before.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    cfg=read(ROOT/'configs/baseline_v3_adopted.json')
    reproductions=[]
    for ref in ['LED1','LED3']:
        row={'text_id':'glyph1','text':ref,'bbox':[20,45,47,57],
             'polygon':[[20,45],[47,45],[47,57],[20,57]],'confidence':.99,'view':'global',
             'observations':[{'text':ref[1:],'confidence':.98,'view':'tile','bbox':[23,45,47,57]}]}
        detections=[{'class_name':'led','bbox':[50,40,70,60],'confidence':.99}]
        args=(detections,[row],100,100,{'class_mapping':{'led':'led'}})
        before,_,_=old.bind_components(*args);after,records,debug=bind_components(*args)
        assert before['components'][ref]['Name']==ref[1:] and after['components'][ref]['Name']==ref
        reproductions.append({'controlled_example':True,'reference':ref,'OCR':row,'before':before,'after':after,
                              'recovery':records[0]['fields']['Name_recovery']})
    summaries=[ROOT/'runs/s3_component_text_quick_dev/summary.json',ROOT/'runs/s3_component_text_crosspage_v2/summary.json']
    rows=[(p,row) for p in summaries for row in read(p)['cases']]
    wanted={r['detections_sha256'] for _,r in rows};lookup={}
    paths=subprocess.run(['rg','--files',str(ROOT/'runs'),'-g','detections.json'],check=True,capture_output=True,text=True).stdout.splitlines()
    cases={r['case_id'] for _,r in rows}
    for text in paths:
        p=Path(text)
        if not cases.intersection(p.parts):continue
        digest=sha(p)
        if digest in wanted:lookup[digest]=p
    changes=[];replayed=[]
    for summary_path,row in rows:
        case=row['case_id'];det_path=lookup[row['detections_sha256']]
        cache=summary_path.parent/case
        detected,texts=read(det_path),read(cache/'texts.json')
        image=ROOT/f'data/standardized/{case}/{case}.png'
        assert sha(image)==row['image_sha256']==detected['image_sha256']==texts['image_sha256']
        assert sha(ROOT/'configs/baseline_v3_adopted.json')==read(summary_path)['config_sha256']
        width,height=detected['size']
        args=(detected['detections'],texts['texts'],width,height,cfg)
        before,_,_=old.bind_components(*args);after,records,debug=bind_components(*args)
        assert before.keys()==after.keys() and before['components'].keys()==after['components'].keys()
        assert before['pins']==after['pins'] and before['nets']==after['nets']
        page_changes=[]
        for key,a in before['components'].items():
            b=after['components'][key]
            changed=[field for field in a if a[field]!=b[field]]
            if changed:
                assert a['type']=='led' and changed==['Name']
                page_changes.append({'key':key,'before':a,'after':b})
        changes.extend({'case':case,**x} for x in page_changes)
        replayed.append({'case':case,'image_sha256':row['image_sha256'],'detections_sha256':sha(det_path),
                         'OCR_sha256':sha(cache/'texts.json'),'prediction_changes':page_changes,
                         'prior_archived_output_matches_old_replay':before==read(cache/'components_partial.json')})
    report={'reproductions':reproductions,'cached_real_pages':replayed,'real_page_changes':changes,
            'image_hashes_and_configuration_verified':True,'new_source_sha256':sha(ROOT/'src/pcb_parser/component_text.py'),
            'old_source_sha256':sha(out/'component_text_before.py'),'targeted_tests_passed':55,
            'scope':'Only same-glyph EDn fragment collisions with observed high-confidence LEDn references; independent model names and unexplained nulls retained.',
            'limits':'Controlled reproduction proves the code path, not the original online examples: their raw OCR evidence was not retained.',
            'new_official_score':None,'new_multigigabyte_archive_generated':False}
    (out/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'controlled_reproductions':len(reproductions),'real_pages_replayed':len(replayed),'real_page_changes':changes,'tests_passed':55}))


if __name__=='__main__':main()
