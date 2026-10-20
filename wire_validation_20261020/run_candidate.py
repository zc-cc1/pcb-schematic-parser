"""Serial CPU replay of each completed baseline case with identical model caches."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/os.environ.get('PCB_VALIDATION_RUNTIME_NAME','runtime_candidate')))
import agent


def main():
    output=HERE/os.environ.get('PCB_VALIDATION_OUTPUT_NAME','candidate_path')
    output.mkdir(exist_ok=True)
    index_path=HERE/(output.name+'_cache_index.json')
    os.environ['PCB_VALIDATION_CACHE_INDEX']=str(index_path)
    agent.configure_environment(output)
    agent.preflight('cpu',output/'runtime_environment.json')
    # The graph experiment does not need another LED detector pass. Freeze that
    # unrelated output from the baseline model, avoiding CPU/GPU detector noise.
    import pcb_parser.led_geometry as led
    baseline_target={}
    def frozen_led(rgb,component_payload,target,predictor):
        target=copy.deepcopy(target)
        for cid,component in target['components'].items():
            if component.get('type')=='led' and cid in baseline_target.get('components',{}):
                component['bbox']=copy.deepcopy(baseline_target['components'][cid]['bbox'])
        return target,{'source':'same-case frozen baseline LED model output; no GT access'}
    led.refine_led_boxes=frozen_led
    led.yolo_predictor=lambda *args,**kwargs: None
    rows=[]
    for image in agent.images_in(Path('D:/pcb_open_gt_in')):
        case=image.stem
        final=HERE/'baseline/final_results'/f'{case}_Pred.json'
        start=time.monotonic()
        while not final.exists():
            if time.monotonic()-start>1800:
                raise TimeoutError('Baseline has not produced '+case)
            time.sleep(2)
        baseline_target.clear()
        baseline_target.update(json.loads(final.read_text(encoding='utf-8')))
        image_hash=hashlib.sha256(image.read_bytes()).hexdigest()
        entries={}
        for work in (HERE/'baseline/work').iterdir():
            pin_folder=work/'p_stage23_stage21_stage20_stage18/stage17/stage16/upstream/upstream'
            meta=work/'p_stage23/probability_inputs.json'
            if not meta.exists() or not (pin_folder/'pin_text_result.json').exists():
                continue
            key=json.loads((pin_folder/'pin_text_result.json').read_text(encoding='utf-8'))['image_sha256']
            paths=sorted((meta.parent/'probabilities').glob('scale_*.npz'))
            if len(paths)==3:
                entries[key]={'pin_folder':str(pin_folder),'capacitor_probabilities':[str(p) for p in paths],
                              'capacitor_metadata':str(meta)}
        assert image_hash in entries, 'Missing complete baseline caches: '+case
        index_path.write_text(json.dumps(entries,indent=2),encoding='utf-8')
        rows.append(agent.run_case(image,output,'cpu','target',True))
        agent.write(output/'validation_progress.json',{'cases':rows,'successes':sum(r['success'] for r in rows)})
    agent.write(output/'decision_log.json',{'method':'isolated path-preserving split replay',
        'device':'cpu; all neural predictions and LED boxes reused from baseline',
        'cases':rows,'successes':sum(r['success'] for r in rows)})
    if not all(r['success'] for r in rows):
        raise SystemExit(1)


if __name__=='__main__':main()
