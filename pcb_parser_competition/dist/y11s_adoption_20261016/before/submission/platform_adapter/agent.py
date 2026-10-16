#!/usr/bin/env python3
"""Competition batch entrypoint. Inference reads PNGs and bundled assets only."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from adapter import COMPONENT_OUTPUT_POLICY, to_sample, to_target, validate_target

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
    for path in [str(ROOT/'src'), str(ROOT/'scripts')]:
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['YOLO_CONFIG_DIR'] = str(output/'runtime/ultralytics')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['MPLCONFIGDIR'] = str(output/'runtime/matplotlib')
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')


def probe_runtime_device(torch, requested):
    """Prefer CUDA only after an actual kernel succeeds, and retain its failure."""
    probe = {'attempted': requested != 'cpu', 'success': False}
    if requested == 'cpu':
        return 'cpu', probe
    candidate = 'cuda:0' if requested == 'auto' else requested
    try:
        if not torch.cuda.is_available():
            # is_available() alone loses the useful driver/visibility error.
            torch.cuda.init()
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA initialization exposed no usable GPU')
        tensor = torch.ones((4, 4), device=candidate)
        value = (tensor @ tensor).sum().item()
        torch.cuda.synchronize(candidate)
        if value != 64:
            raise RuntimeError('CUDA matrix probe returned an unexpected result')
        probe.update(success=True, device=candidate, name=torch.cuda.get_device_name(candidate),
                     capability=list(torch.cuda.get_device_capability(candidate)),
                     compiled_arches=torch.cuda.get_arch_list())
    except Exception as error:
        probe['error'] = f'{type(error).__name__}: {error}'
    return ('cuda:0' if probe['success'] else 'cpu') if requested == 'auto' else requested, probe


def preflight(device, diagnostics_path=None):
    import torch
    manifest = json.loads((ROOT/'package_manifest.json').read_text(encoding='utf-8'))
    for relative in manifest['runtime_assets']:
        path = ROOT/relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != manifest['files'][relative]['sha256']:
            raise ValueError(f'Bundled asset hash differs: {relative}')
    cuda_available = torch.cuda.is_available()
    diagnostics = {'requested_device':device, 'torch':torch.__version__, 'torch_cuda_build':torch.version.cuda,
                   'cuda_available':cuda_available, 'cuda_device_count':torch.cuda.device_count(),
                   'environment':{key:os.environ.get(key) for key in ['PCB_DEVICE','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','NVIDIA_DRIVER_CAPABILITIES']}}
    device, diagnostics['cuda_probe'] = probe_runtime_device(torch, device)
    diagnostics['selected_device'] = device
    if diagnostics_path:
        write(diagnostics_path, diagnostics)
    print('[Submission] runtime=' + json.dumps(diagnostics, ensure_ascii=True), flush=True)
    if device.startswith('cuda') and not diagnostics['cuda_probe']['success']:
        raise RuntimeError('CUDA requested but unusable: ' + diagnostics['cuda_probe'].get('error', 'unknown error'))
    if device == 'cpu' and diagnostics['cuda_probe']['attempted']:
        print('[Submission] CUDA unavailable; using CPU. Reason: ' + diagnostics['cuda_probe']['error'], flush=True)
    print(f'[Submission] device={device}; output_format={os.environ.get("PCB_OUTPUT_FORMAT", "target")}', flush=True)
    return device


def choose_workers(requested, device, case_count, gpu_memory_bytes=0):
    if requested < 0 or requested > 3:
        raise ValueError('workers must be 0 (auto), 1, 2 or 3')
    selected = requested or (3 if device.startswith('cuda') and gpu_memory_bytes >= 24*1024**3 else 1)
    return min(selected, case_count)


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
        child_environment = os.environ.copy()
        # Parallel cases must not race on Ultralytics/Matplotlib settings files.
        child_environment['YOLO_CONFIG_DIR'] = str(output/'runtime/ultralytics'/case_id)
        child_environment['MPLCONFIGDIR'] = str(output/'runtime/matplotlib'/case_id)
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.run(command, cwd=ROOT, env=child_environment,
                                     stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f'Image pipeline exited with code {process.returncode}; inspect {log.name}')
        paper = json.loads((stage/'export/engineering_submission.json').read_text(encoding='utf-8'))
        from component_export import export_power_components, missing_field_diagnostics
        records_document = json.loads((stage/'export/completed_component_records.json').read_text(encoding='utf-8'))
        summary = json.loads((stage/'summary.json').read_text(encoding='utf-8'))
        ocr_paths = [Path(p) for p in summary['original_S23_input_sha256'] if p.endswith('texts.json')]
        if len(ocr_paths) != 1:
            raise ValueError('Expected one current-image OCR provenance file')
        ocr_document = json.loads(ocr_paths[0].read_text(encoding='utf-8'))
        component_payload, component_audit = export_power_components(paper, records_document, ocr_document)
        target = to_target(component_payload)
        from pcb_parser.ic_name_selection import select_ic_names
        target, name_audit = select_ic_names(target, records_document, ocr_document)
        validate_target(target)
        write(output/'logs'/f'{case_id}_ic_name_selection.json', name_audit)
        write(output/'logs'/f'{case_id}_ocr_texts.json', ocr_document)
        rgb = None
        if output_format == 'target' and any(c['type'] in {'ic','box'} for c in target['components'].values()):
            ic_start = time.perf_counter()
            from PIL import Image
            import numpy as np
            from pcb_parser.ic_geometry import refine_ic_boxes
            rgb = np.array(Image.open(image).convert('RGB'))
            target, ic_audit = refine_ic_boxes(rgb, target)
            validate_target(target)
            ic_audit['image_sha256'] = summary['image_sha256']
            ic_audit['wall_seconds'] = round(time.perf_counter()-ic_start, 3)
            row['ic_geometry_seconds'] = ic_audit['wall_seconds']
            write(output/'logs'/f'{case_id}_ic_geometry.json', ic_audit)
        if output_format == 'target' and any(c['type']=='led' for c in component_payload['components'].values()):
            led_start = time.perf_counter()
            from PIL import Image
            import numpy as np
            from pcb_parser.led_geometry import refine_led_boxes, yolo_predictor
            if rgb is None:
                rgb = np.array(Image.open(image).convert('RGB'))
            target, led_audit = refine_led_boxes(rgb, component_payload, target,
                yolo_predictor(ROOT/'weights/detector_dataset300_v3_nano.pt', device))
            validate_target(target)
            led_audit['image_sha256'] = summary['image_sha256']
            led_audit['wall_seconds'] = round(time.perf_counter()-led_start, 3)
            row['led_geometry_seconds'] = led_audit['wall_seconds']
            write(output/'logs'/f'{case_id}_led_geometry.json', led_audit)
        write(output/'project_results'/f'{case_id}_Pred.json', paper)
        write(output/'component_results'/f'{case_id}_Pred.json', component_payload)
        write(output/'logs'/f'{case_id}_component_export_audit.json', component_audit)
        write(output/'logs'/f'{case_id}_component_fields_debug.json', missing_field_diagnostics(records_document, ocr_document))
        prediction = to_sample(paper) if output_format == 'sample' else (paper if output_format == 'project' else target)
        write(output/'final_results'/f'{case_id}_Pred.json', prediction)
        for name in ['engineering_export_coverage.json', 'field_provenance.json']:
            shutil.copy2(stage/'export'/name, output/'logs'/f'{case_id}_{name}')
        shutil.copy2(stage/'summary.json', output/'logs'/f'{case_id}_summary.json')
        row.update(success=True, components=len(target['components']),
                   pins=sum(len(pins) for pins in target['pins'].values()), nets=len(target['nets']))
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
    parser.add_argument('--output-format', choices=['target', 'sample', 'project'], default=os.environ.get('PCB_OUTPUT_FORMAT', 'target'))
    parser.add_argument('--workers', type=int, choices=[0, 1, 2, 3], default=int(os.environ.get('PCB_WORKERS', '0')))
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
        device = preflight(args.device, args.output/'runtime_environment.json')
        gpu_memory = 0
        if device.startswith('cuda'):
            import torch
            gpu_memory = torch.cuda.get_device_properties(device).total_memory
        workers = choose_workers(args.workers, device, len(images), gpu_memory)
        decision.update(device=device, input_cases=len(images), workers=workers,
                        gpu_memory_bytes=gpu_memory, protocol='2026-10-08-global-power-rails',
                        type_mapping_status='six aliases verified in submission 8180 and public labels',
                        component_output_policy=COMPONENT_OUTPUT_POLICY)
        print(f'[Submission] workers={workers}; gpu_memory_bytes={gpu_memory}', flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_case, image, args.output, device, args.output_format, args.keep_work)
                       for image in images]
            completed = (future.result() for future in as_completed(futures))
            for row in completed:
                decision['cases'].append(row)
                decision['processed_cases'] += 1
                decision['successful_cases' if row['success'] else 'failed_cases'] += 1
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
