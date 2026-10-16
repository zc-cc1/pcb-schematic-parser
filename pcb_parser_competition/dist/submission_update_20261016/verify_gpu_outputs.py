"""Verify the local GPU run of the frozen source and write gpu_verification.json.

PowerShell 5.1's ConvertFrom-Json rejects the selection/geometry audit files
(they contain repeated keys), so the audit is parsed here instead.

Usage: verify_gpu_outputs.py <gpu-output-dir> <gpu_verification.json>
"""
import json
import sys
from pathlib import Path

CASES = ['0036', '0147', '0107']
POLICY_MARKERS = ['led_geometry_v1', 'ic_name_priority_v1', 'ic_geometry_v1']


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise SystemExit('FAILED: ' + message)


def main():
    output = Path(sys.argv[1])
    summary_path = Path(sys.argv[2])
    decision = load(output / 'final_results' / 'decision_log.json')
    runtime = load(output / 'runtime_environment.json')

    require(decision['successful_cases'] == len(CASES), f"expected {len(CASES)} successful cases")
    require(decision['failed_cases'] == 0, 'no case may fail')
    require(decision['output_format'] == 'target', 'deployment must emit the Target protocol')
    require(decision['device'].startswith('cuda'), f"device is {decision['device']}, not CUDA")
    require(runtime['cuda_probe']['success'] is True, 'the CUDA matrix probe must succeed')
    require(runtime['cuda_probe']['attempted'] is True, 'the probe must actually be attempted')
    require(runtime['selected_device'].startswith('cuda'), 'auto-selection must land on CUDA')
    require(runtime['cuda_available'] is True and runtime['cuda_device_count'] >= 1, 'CUDA must be visible')
    require(runtime['requested_device'] == 'auto', 'the deployed default must stay auto')

    policy = decision['component_output_policy']
    for marker in POLICY_MARKERS:
        require(marker in policy, f'policy does not carry {marker}')

    # The IC stages always run for these pages; the LED stage only runs when the
    # page actually contains an LED, which the run log reports per case.
    required = {}
    for row in decision['cases']:
        case = row['case_id']
        required[case] = [f'{case}_ic_name_selection.json', f'{case}_ic_geometry.json']
        if 'led_geometry_seconds' in row:
            required[case].append(f'{case}_led_geometry.json')
    logs = {}
    for case, names in required.items():
        for name in names:
            path = output / 'logs' / name
            require(path.exists(), f'missing fix audit log: {name}')
            logs[f'{case}/{name}'] = len(load(path)['entries' if name.endswith('_geometry.json') else 'components'])
    led_cases = [row['case_id'] for row in decision['cases'] if 'led_geometry_seconds' in row]
    require(led_cases, 'no case exercised the LED geometry stage')

    led = load(output / 'logs' / '0107_led_geometry.json')
    led_reasons = sorted({entry['axis_evidence']['reason'] for entry in led['entries'] if entry.get('axis_evidence')} |
                         {entry['local_detection']['reason'] for entry in led['entries'] if entry.get('local_detection')})
    ic = load(output / 'logs' / '0036_ic_geometry.json')
    ic_changed = sum(1 for entry in ic['entries'] if entry.get('changed'))

    summary = {
        'image': 'pcb-parser:local-cuda-68d6bcb8',
        'image_torch': runtime['torch'],
        'image_torch_cuda_build': runtime['torch_cuda_build'],
        'gpu_network_disabled': True,
        'source_read_only_mounted': True,
        'runs_scratch': 'tmpfs at /submission/runs (required by the Ultralytics predictor save_dir)',
        'cases': CASES,
        'successful_cases': decision['successful_cases'],
        'failed_cases': decision['failed_cases'],
        'requested_device': runtime['requested_device'],
        'selected_device': runtime['selected_device'],
        'cuda_available': runtime['cuda_available'],
        'cuda_device_count': runtime['cuda_device_count'],
        'cuda_probe': runtime['cuda_probe'],
        'device_capability_probed': runtime['cuda_probe']['capability'],
        'compiled_arches': runtime['cuda_probe']['compiled_arches'],
        'component_output_policy': policy,
        'output_format': decision['output_format'],
        'wall_seconds': decision['wall_seconds'],
        'fix_audit_rows': logs,
        'led_stage_cases': led_cases,
        'ic_geometry_changed_0036': ic_changed,
        'led_geometry_changed_boxes_0107': led['changed_LED_boxes'],
        'led_geometry_changed_centers_0107': led['changed_LED_centers'],
        'led_geometry_reasons_0107': led_reasons,
        'led_geometry_wall_seconds_0107': led['wall_seconds'],
        'case_results': [
            {key: row[key] for key in ('case_id', 'success', 'components', 'pins', 'nets', 'seconds')
             if key in row} | {k: v for k, v in row.items() if k.endswith('_geometry_seconds')}
            for row in decision['cases']
        ],
        'scope': ('Local RTX 5060 (sm_120) with the frozen source bind-mounted read-only. Proves automatic '
                  'CUDA selection, real GPU inference and the IC/LED fix logs for this exact source. It is NOT '
                  'an A800/cu121 measurement of the uploaded archive.'),
        'new_multigigabyte_archive_generated': False,
        'official_score': None,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"GPU verification PASSED: device={summary['selected_device']} on {summary['cuda_probe']['name']}, "
          f"{summary['successful_cases']}/{len(CASES)} cases, wall={summary['wall_seconds']}s")


if __name__ == '__main__':
    main()