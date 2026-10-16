#!/usr/bin/env python3
"""Competition batch entrypoint. Inference reads PNGs and bundled assets only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

from adapter import to_sample

ROOT = Path(__file__).resolve().parent
EMPTY = {'components': {}, 'pins': {}, 'nets': {}}


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def images_in(directory):
    images = sorted((p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() == '.png'),
                    key=lambda p: p.relative_to(directory).as_posix())
    ids = [p.stem.casefold() for p in images]
    if len(ids) != len(set(ids)):
        raise ValueError('Input contains repeated image stems; case_id output files would collide')
    if not images:
        raise ValueError(f'No PNG images found under {directory}')
    return images


def configure_environment(output):
    os.environ['PYTHONPATH'] = os.pathsep.join([str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT)])
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['YOLO_CONFIG_DIR'] = str(output/'runtime/ultralytics')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['MPLCONFIGDIR'] = str(output/'runtime/matplotlib')
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')


def preflight(device):
    import torch
    manifest = json.loads((ROOT/'package_manifest.json').read_text(encoding='utf-8'))
    for relative in manifest['runtime_assets']:
        path = ROOT/relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != manifest['files'][relative]['sha256']:
            raise ValueError(f'Bundled asset hash differs: {relative}')
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    print(f'[Submission] device={device}; output_format={os.environ.get("PCB_OUTPUT_FORMAT", "sample")}', flush=True)
    return device


def run_case(image, output, device, output_format, keep_work):
    case_id = image.stem
    row = {'case_id': case_id, 'input': str(image), 'success': False}
    start = time.perf_counter()
    log = output/'logs'/f'{case_id}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    work_parent = output/'work' if keep_work else None
    if work_parent:
        work_parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix='pcb_', dir=work_parent))
    else:
        work = Path(tempfile.mkdtemp(prefix='pcb_'))
    try:
        stage = work/'p'
        command = [sys.executable, str(ROOT/'scripts/run_observed_net_fields.py'),
                   '--image', str(image), '--output', str(stage), '--device', device]
        print(f'[Submission] Start {case_id}; log={log}', flush=True)
        # Stream child output to a durable file; no unbounded in-memory log buffer.
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f'Image pipeline exited with code {process.returncode}; inspect {log.name}')
        paper = json.loads((stage/'export/engineering_submission.json').read_text(encoding='utf-8'))
        # Validate the sample mapping even when exporting the original paper layout.
        sample = to_sample(paper)
        write(output/'project_results'/f'{case_id}_Pred.json', paper)
        write(output/'final_results'/f'{case_id}_Pred.json', sample if output_format == 'sample' else paper)
        for name in ['engineering_export_coverage.json', 'field_provenance.json']:
            shutil.copy2(stage/'export'/name, output/'logs'/f'{case_id}_{name}')
        shutil.copy2(stage/'summary.json', output/'logs'/f'{case_id}_summary.json')
        row.update(success=True, components=len(sample['components']), pins=len(sample['pins']), nets=len(sample['nets']))
    except Exception as error:
        row['error'] = f'{type(error).__name__}: {error}'
        with log.open('a', encoding='utf-8') as stream:
            traceback.print_exc(file=stream)
        write(output/'final_results'/f'{case_id}_Pred.json', EMPTY)
        print(f'[Submission] FAILED {case_id}: {row["error"]}', flush=True)
        if log.exists():
            print('\n'.join(log.read_text(encoding='utf-8', errors='replace').splitlines()[-24:]), flush=True)
    finally:
        row['seconds'] = round(time.perf_counter()-start, 3)
        if not keep_work:
            shutil.rmtree(work)
    print(f'[Submission] End {case_id}: {json.dumps(row, ensure_ascii=True)}', flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', '--input', type=Path, required=True)
    parser.add_argument('-output', '--output', type=Path, required=True)
    parser.add_argument('--device', default=os.environ.get('PCB_DEVICE', 'auto'))
    parser.add_argument('--output-format', choices=['sample', 'project'], default=os.environ.get('PCB_OUTPUT_FORMAT', 'sample'))
    parser.add_argument('--keep-work', action='store_true')
    args = parser.parse_args()
    args.input, args.output = args.input.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.input == args.output or args.output.is_relative_to(args.input) or args.input.is_relative_to(args.output):
        parser.error('Input and output directories must not overlap')
    configure_environment(args.output)
    start = time.perf_counter()
    decision = {'method': 'yolov8_v3_rapidocr_unet_v1_S24_engineering',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'processed_cases': 0, 'successful_cases': 0, 'failed_cases': 0,
                'output_format': args.output_format, 'coordinate_origin': 'bottom_left_H_minus_y',
                'token_usage': {'total_tokens': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'api_calls': 0},
                'cases': []}
    fatal = False
    try:
        if not args.input.is_dir():
            raise FileNotFoundError(args.input)
        images = images_in(args.input)
        device = preflight(args.device)
        decision.update(device=device, input_cases=len(images))
        for image in images:
            row = run_case(image, args.output, device, args.output_format, args.keep_work)
            decision['cases'].append(row)
            decision['processed_cases'] += 1
            decision['successful_cases' if row['success'] else 'failed_cases'] += 1
            # Both documented locations are written because the README and live
            # platform instructions disagree about the decision-log directory.
            decision['wall_seconds'] = round(time.perf_counter()-start, 3)
            write(args.output/'final_results/decision_log.json', decision)
            write(args.output/'decision_log.json', decision)
    except Exception as error:
        fatal = True
        decision['fatal_error'] = f'{type(error).__name__}: {error}'
        traceback.print_exc()
    finally:
        decision['wall_seconds'] = round(time.perf_counter()-start, 3)
        write(args.output/'final_results/decision_log.json', decision)
        write(args.output/'decision_log.json', decision)
    return 1 if fatal or not decision['successful_cases'] else 0


if __name__ == '__main__':
    sys.exit(main())
