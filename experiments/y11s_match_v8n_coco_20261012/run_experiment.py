"""Train a COCO-initialized YOLO11s with the adopted v8n final-stage recipe.

Original source project and adopted weights are read-only inputs. Outputs stay
in this experiment directory. Training completion automatically triggers a
fresh, paired deployment evaluation of YOLO11s and the adopted YOLOv8n.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = Path(r'C:\Users\Pc\Desktop\新建文件夹\pcb_parser_competition')
PYTHON = Path(r'C:\Users\Pc\Desktop\vision\pcb_parser_competition\.venv-gpu\Scripts\python.exe')
CONFIG = HERE / 'experiment_config.json'
os.environ['YOLO_CONFIG_DIR'] = str(HERE / 'ultralytics')
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    temp.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def libraries():
    (HERE / 'ultralytics').mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(SOURCE / 'scripts'))
    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    settings.update({'sync': False})
    return torch, ultralytics, YOLO


def prepare():
    torch, ultralytics, _ = libraries()
    source_record = read(SOURCE / 'runs/training/detector_dataset300_v3_nano_record.json')
    source_config = read(SOURCE / 'configs/detector_training_v3.json')
    assert source_record['effective_training'] == source_config['training']
    dataset = SOURCE / source_config['dataset']
    manifest = dataset.parent / 'manifest.jsonl'
    assert sha(manifest) == source_record['dataset_manifest_sha256']
    assert sha(dataset) == source_record['data_config_sha256']
    rows = [json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines()]
    groups_train = {r['group'] for r in rows if r['split'] == 'train'}
    groups_val = {r['group'] for r in rows if r['split'] == 'val'}
    assert not groups_train & groups_val, 'Source groups overlap'
    for i, row in enumerate(rows):
        for kind in ('image', 'label'):
            assert sha(row[kind]) == row[kind + '_sha256'], f"Changed input: {row['case_id']} {kind}"
        if i % 100 == 0:
            print(json.dumps({'preflight_inputs_verified': i + 1, 'total': len(rows)}), flush=True)
    official = SOURCE / 'weights/yolo11s.pt'
    provenance = read(SOURCE / 'weights/yolo11s.pt_provenance.json')
    adopted = SOURCE / 'weights/detector_dataset300_v3_nano.pt'
    assert sha(adopted) == source_record['best_weights_sha256']
    cfg = {
        'experiment': 'y11s_match_v8n_coco_20261012',
        'user_choice': 'COCO-initialized YOLO11s; same final-stage training configuration as adopted YOLOv8n',
        'architecture': 'yolo11s.yaml',
        'initial_weights': str(official),
        'initial_weights_sha256': sha(official),
        'initial_weights_provenance': provenance,
        'dataset': str(dataset),
        'dataset_manifest_sha256': sha(manifest),
        'dataset_yaml_sha256': sha(dataset),
        'training': source_record['effective_training'],
        'baseline_training_record_sha256': sha(SOURCE / 'runs/training/detector_dataset300_v3_nano_record.json'),
        'baseline_weights': str(adopted),
        'baseline_weights_sha256': sha(adopted),
        'baseline_evaluation': str(SOURCE / 'runs/eval_v3_fused_v2thr.json'),
        'eval_manifest': str(SOURCE / 'data/detection_v1/manifest.jsonl'),
        'eval_config': str(SOURCE / 'configs/baseline_v3.json'),
        'image_counts': dict(Counter(r['split'] for r in rows)),
        'runtime': {'python': sys.executable, 'torch': torch.__version__, 'ultralytics': ultralytics.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        'comparability_limit': 'Same final-stage settings/data; baseline v8n has prior task-specific warm starts and new YOLO11s starts from COCO. This is not an isolated architecture comparison.',
        'evaluation_policy': 'Same 59-page bbox/category protocol. Report global conf 0.2, adopted baseline thresholds, and symmetric per-class retuning at fixed inference conf 0.2. Retuned validation F1 is exploratory, not independent test F1.',
        'checkpoint_policy': 'Use Ultralytics best.pt selected by mAP50-95, as for the adopted v8n. Also report last.pt. Keep both weights; do not automatically change deployment.',
        'created_at_utc': utc(),
    }
    assert torch.cuda.is_available(), 'CUDA unavailable'
    write(CONFIG, cfg)
    write(HERE / 'status.json', {'status': 'prepared', 'checked_inputs': len(rows), 'config': str(CONFIG), 'updated_at_utc': utc()})
    print(json.dumps({'status': 'prepared', 'config': str(CONFIG), 'training': cfg['training'], 'gpu': cfg['runtime']['gpu']}, ensure_ascii=False), flush=True)


def evaluate(collected, thresholds):
    from run_baseline import apply_class_confidence
    from evaluate_detector import counts_for_case, metric
    total = Counter()
    classes = {}
    cases = []
    for case_id, predictions, truth in collected:
        counts = counts_for_case(apply_class_confidence(predictions, thresholds), truth, .5)
        subtotal = Counter()
        for name, value in counts.items():
            subtotal.update(value)
            classes.setdefault(name, Counter()).update(value)
        total.update(subtotal)
        cases.append({'case_id': case_id, 'counts': dict(subtotal)})
    return {'overall': metric(total), 'by_class': {k: metric(v) for k, v in classes.items()}, 'cases': cases}


def tune(collected):
    from tune_class_thresholds import class_counts, prf
    names = sorted({p['class_name'] for _, predictions, truth in collected for p in predictions + truth})
    thresholds = {}
    for name in names:
        scores = {float(p['confidence']) for _, predictions, _ in collected for p in predictions if p['class_name'] == name}
        best_f1 = -1.
        best_t = .2
        for threshold in sorted(scores | {.2, .25, 1.000001}):
            counts = [class_counts(predictions, truth, name, threshold, .5) for _, predictions, truth in collected]
            tp, fp, fn = (sum(c[i] for c in counts) for i in range(3))
            f1 = prf(tp, fp, fn)[2]
            if f1 > best_f1 + 1e-12:
                best_f1, best_t = f1, threshold
        thresholds[name] = best_t
    return thresholds


def compare(cfg, run_dir):
    torch, _, YOLO = libraries()
    import numpy as np
    from PIL import Image
    from sahi import AutoDetectionModel
    from run_baseline import fused_predictions
    from evaluate_detector import metric
    infer_cfg = read(cfg['eval_config'])
    assert infer_cfg['detector']['confidence'] == .2
    rows = [json.loads(s) for s in Path(cfg['eval_manifest']).read_text(encoding='utf-8').splitlines()]
    rows = [r for r in rows if r['split'] == 'val']
    baseline_eval = read(cfg['baseline_evaluation'])
    assert sha(cfg['eval_manifest']) == baseline_eval['manifest_sha256']
    assert {r['case_id'] for r in rows} == {r['case_id'] for r in baseline_eval['cases']}
    models = {'v8n_adopted': Path(cfg['baseline_weights']), 'y11s_best': run_dir/'weights/best.pt', 'y11s_last': run_dir/'weights/last.pt'}
    result = {'status': 'completed', 'cases': len(rows), 'definition': baseline_eval['definition'], 'official_competition_f1': False, 'inference_config': infer_cfg, 'manifest_sha256': sha(cfg['eval_manifest']), 'comparability_limit': cfg['comparability_limit'], 'retuned_results_caveat': 'Thresholds chosen and scored on the same validation set; exploratory, not independent test performance.', 'models': {}}
    for name, weights in models.items():
        write(HERE/'status.json', {'status': 'evaluating', 'model': name, 'updated_at_utc': utc(), 'run_directory': str(run_dir)})
        whole = YOLO(str(weights))
        sliced = AutoDetectionModel.from_pretrained(model_type='ultralytics', model_path=str(weights), confidence_threshold=.2, device='cuda:0', image_size=1536)
        sliced.model.overrides.update({'iou': .5, 'max_det': 1000, 'half': False})
        warm_rgb = np.asarray(Image.open(rows[0]['image']).convert('RGB'))
        fused_predictions(whole, sliced, warm_rgb, 'cuda:0', infer_cfg)
        torch.cuda.synchronize()
        collected = []
        timings = []
        for row in rows:
            rgb = np.asarray(Image.open(row['image']).convert('RGB'))
            torch.cuda.synchronize()
            start = time.perf_counter()
            predictions = fused_predictions(whole, sliced, rgb, 'cuda:0', infer_cfg)
            torch.cuda.synchronize()
            timings.append((time.perf_counter()-start)*1000)
            collected.append((row['case_id'], predictions, row['objects']))
        write(HERE/'evaluation'/f'{name}_predictions.json', collected)
        thresholds = tune(collected)
        entry = {'weights': str(weights), 'sha256': sha(weights), 'global_conf_0_2': evaluate(collected, {}), 'retuned_same_validation': evaluate(collected, thresholds), 'retuned_thresholds': thresholds, 'deployment_latency_ms': {'mean': float(np.mean(timings)), 'median': float(np.median(timings)), 'p95': float(np.quantile(timings, .95)), 'warmup_images': 1, 'timed_images': len(rows), 'scope': 'Preprocessing and fused whole/sliced inference; excludes image I/O, model load, and threshold search.'}}
        if name == 'v8n_adopted':
            entry['adopted_thresholds'] = evaluate(collected, baseline_eval['class_thresholds'])
            entry['original_counts_reproduced'] = all(entry['adopted_thresholds']['overall'][k] == baseline_eval['overall'][k] for k in ('tp','fp','fn'))
            assert entry['original_counts_reproduced'], 'Baseline reproduction failed; comparison cannot be treated as same protocol'
        result['models'][name] = entry
        write(HERE/'comparison.json', result)
        print(json.dumps({'evaluation_model': name, 'global': entry['global_conf_0_2']['overall'], 'retuned': entry['retuned_same_validation']['overall']}, ensure_ascii=False), flush=True)
        del whole, sliced
        torch.cuda.empty_cache()
    baseline = result['models']['v8n_adopted']
    best = result['models']['y11s_best']
    delta = best['retuned_same_validation']['overall']['f1'] - baseline['retuned_same_validation']['overall']['f1']
    result['best_delta_retuned_f1'] = delta
    result['best_delta_global_f1'] = best['global_conf_0_2']['overall']['f1'] - baseline['global_conf_0_2']['overall']['f1']
    a = {c['case_id']:c['counts'] for c in baseline['retuned_same_validation']['cases']}
    b = {c['case_id']:c['counts'] for c in best['retuned_same_validation']['cases']}
    ids = sorted(a)
    av = np.array([[a[x][k] for k in ('tp','fp','fn')] for x in ids])
    bv = np.array([[b[x][k] for k in ('tp','fp','fn')] for x in ids])
    indices = np.random.default_rng(20261012).integers(0,len(ids),size=(20000,len(ids)))
    aa, bb = av[indices].sum(1), bv[indices].sum(1)
    diffs = 2*bb[:,0]/(2*bb[:,0]+bb[:,1]+bb[:,2])-2*aa[:,0]/(2*aa[:,0]+aa[:,1]+aa[:,2])
    result['paired_page_bootstrap'] = {'ci95_delta_f1': np.quantile(diffs,[.025,.975]).tolist(), 'resamples': 20000, 'caveat': 'Does not correct threshold tuning or repeated validation selection bias.'}
    result['completed_at_utc'] = utc()
    write(HERE/'comparison.json', result)
    lines = ['# YOLO11s 相同末阶段配置训练与对比', '', 'YOLO11s 从 COCO 权重重新训练；现有 v8n 有任务热启动历史。因此参数对齐，但初始化历史不同。以下是本地类别+bbox F1，不是官方 ComponentF1。', '', '| 模型 | 全局 conf 0.2 F1 | 相同方法逐类重调 F1 | P / R（重调） |', '|---|---:|---:|---|']
    for name, entry in result['models'].items():
        m = entry['retuned_same_validation']['overall']
        lines.append(f"| {name} | {entry['global_conf_0_2']['overall']['f1']:.6f} | {m['f1']:.6f} | {m['precision']:.4f} / {m['recall']:.4f} |")
    lines += ['', f'YOLO11s best 与 v8n 的同法重调 F1 差：{delta:+.6f}（{delta*100:+.3f} 个百分点）。', '', '重调在同一验证集上选择并计分，存在选择偏差；paired bootstrap 区间也没有修正此偏差。部署模型未自动替换。', '', '## 逐类差异（best，相同阈值搜索方法）', '', '| 类别 | 真值数 | v8n F1 | YOLO11s F1 | Δ TP | Δ FP |', '|---|---:|---:|---:|---:|---:|']
    ca, cb = baseline['retuned_same_validation']['by_class'], best['retuned_same_validation']['by_class']
    for name in sorted(set(ca)|set(cb)):
        x, y = ca.get(name, metric({})), cb.get(name, metric({}))
        lines.append(f"| {name} | {x.get('tp',0)+x.get('fn',0)} | {x['f1']:.4f} | {y['f1']:.4f} | {y.get('tp',0)-x.get('tp',0):+d} | {y.get('fp',0)-x.get('fp',0):+d} |")
    lines += ['', '详细配置、权重哈希、逐页计数、阈值和推理耗时见 experiment_config.json、training_record.json 和 comparison.json。']
    (HERE/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def train_and_compare(resume=False):
    torch, ultralytics, YOLO = libraries()
    from train_detector import ClassNameTransferTrainer
    cfg = read(CONFIG)
    assert sha(cfg['initial_weights']) == cfg['initial_weights_sha256']
    assert sha(cfg['baseline_weights']) == cfg['baseline_weights_sha256']
    run_dir = HERE/'runs'/'y11s_coco_matched_v8n'
    started = time.perf_counter()
    record = read(HERE/'training_record.json') if resume else {'status':'training','started_at_utc':utc(),'configuration':cfg,'python':sys.executable}
    record['status'] = 'training'
    write(HERE/'training_record.json', record)
    model = YOLO(str(run_dir/'weights/last.pt')) if resume else YOLO(cfg['architecture'])
    if not resume:
        model.load(cfg['initial_weights'])
    def progress(trainer):
        state = {'status':'training','epoch_completed':trainer.epoch+1,'epochs_planned':trainer.epochs,'seconds_this_process':time.perf_counter()-started,'metrics':{k:float(v) for k,v in trainer.metrics.items()},'run_directory':str(trainer.save_dir),'updated_at_utc':utc(),'pid':os.getpid()}
        write(HERE/'status.json', state)
        print(json.dumps({'epoch_completed':state['epoch_completed'],'epochs_planned':state['epochs_planned'],'mAP50':state['metrics'].get('metrics/mAP50(B)'),'elapsed':state['seconds_this_process']}),flush=True)
    def train_start(trainer):
        actual = vars(trainer.args)
        # Ultralytics normalizes CUDA device 0 to the equivalent string "0".
        def equivalent(key, expected, found):
            return str(expected) == str(found) if key == 'device' else expected == found
        mismatches = {k:{'expected':v,'actual':actual.get(k)} for k,v in cfg['training'].items() if not equivalent(k,v,actual.get(k))}
        write(HERE/'effective_args.json', actual)
        assert not mismatches, f'Effective arguments changed: {mismatches}'
        record['run_directory'] = str(trainer.save_dir)
        record['parameter_count'] = sum(p.numel() for p in trainer.model.parameters())
        record['effective_arguments_verified'] = True
        write(HERE/'training_record.json', record)
        write(HERE/'status.json', {'status':'training','epoch_completed':getattr(trainer,'start_epoch',0),'epochs_planned':trainer.epochs,'run_directory':str(trainer.save_dir),'updated_at_utc':utc(),'pid':os.getpid()})
    model.add_callback('on_fit_epoch_end',progress)
    model.add_callback('on_train_start',train_start)
    try:
        if resume:
            model.train(resume=True)
        else:
            model.train(trainer=ClassNameTransferTrainer,data=cfg['dataset'],project=str(HERE/'runs'),name='y11s_coco_matched_v8n',exist_ok=False,**cfg['training'])
        run_dir = Path(model.trainer.save_dir)
        record.update(status='trained',finished_at_utc=utc(),seconds_this_process=time.perf_counter()-started,run_directory=str(run_dir),best_weights_sha256=sha(run_dir/'weights/best.pt'),last_weights_sha256=sha(run_dir/'weights/last.pt'))
        write(HERE/'training_record.json',record)
        del model
        torch.cuda.empty_cache()
        compare(cfg,run_dir)
        write(HERE/'status.json',{'status':'completed','completed_at_utc':utc(),'comparison':str(HERE/'comparison.md'),'run_directory':str(run_dir),'pid':os.getpid()})
    except BaseException as exc:
        record.update(status='failed',failed_at_utc=utc(),error=repr(exc))
        write(HERE/'training_record.json',record)
        write(HERE/'status.json',{'status':'failed','error':repr(exc),'updated_at_utc':utc(),'pid':os.getpid()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--evaluate-only',action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.evaluate_only:
        libraries()
        compare(read(CONFIG),HERE/'runs'/'y11s_coco_matched_v8n')
    else:
        train_and_compare(args.resume)
