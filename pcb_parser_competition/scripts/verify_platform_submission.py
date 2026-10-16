"""Extract the actual ZIP and execute it against PNG-only smoke-test inputs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    tick = time.perf_counter()
    archive = ROOT/'dist/submission.zip'
    sequence = 1
    while (ROOT/f'tmp/platform_smoke_{sequence}').exists():
        sequence += 1
    workspace = ROOT/f'tmp/platform_smoke_{sequence}'
    package, input_dir, output = workspace/'package', workspace/'input', workspace/'output'
    input_dir.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            destination = (package/name).resolve()
            if not destination.is_relative_to(package.resolve()):
                raise ValueError(f'Unsafe ZIP member: {name}')
        bundle.extractall(package)
    manifest = read(package/'package_manifest.json')
    for name, expected in manifest['files'].items():
        path = package/name
        assert path.stat().st_size == expected['bytes']
        assert sha(path) == expected['sha256'], name
    # A nested PNG checks recursive scanning; no annotation/reference JSON copied.
    cases = ['0036', '0042']
    input_hashes = {}
    for index, case in enumerate(cases):
        target = input_dir/(f'nested/{case}.png' if index else f'{case}.png')
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT/f'data/standardized/{case}/{case}.png', target)
        input_hashes[case] = sha(target)
    assert [p.suffix for p in input_dir.rglob('*') if p.is_file()] == ['.png', '.png']
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    log = ROOT/'dist/local_smoke.log'
    command = [sys.executable, str(package/'agent.py'), '-input', str(input_dir),
               '-output', str(output), '--device', 'cpu']
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=workspace, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text(encoding='utf-8', errors='replace')[-16000:])
        raise RuntimeError(f'Extracted ZIP inference failed: {result.returncode}')
    decision = read(output/'final_results/decision_log.json')
    assert decision == read(output/'decision_log.json')
    assert decision['processed_cases'] == len(cases)
    assert decision['successful_cases'] == len(cases) and decision['failed_cases'] == 0
    assert all(value == 0 for value in decision['token_usage'].values())
    spec = importlib.util.spec_from_file_location('extracted_adapter', package/'adapter.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    expected_names = {f'{case}_Pred.json' for case in cases} | {'decision_log.json'}
    assert {p.name for p in (output/'final_results').iterdir()} == expected_names
    counts = []
    for case in cases:
        prediction = read(output/f'final_results/{case}_Pred.json')
        project = read(output/f'project_results/{case}_Pred.json')
        assert prediction == module.to_sample(project)
        assert prediction['components'], f'Empty component output for {case}'
        for cid, component in prediction['components'].items():
            assert component['bbox'] == project['components'][cid]['bbox']
        for net in prediction['nets'].values():
            assert set(net['pins']) <= set(prediction['pins'])
            assert all(len(segment['points']) == 2 for segment in net['wire_segments'])
        counts.append({'case_id': case, 'components': len(prediction['components']),
                       'pins': len(prediction['pins']), 'nets': len(prediction['nets'])})
    assert any(row['nets'] > 0 for row in counts), 'No complete network exercised by the smoke test'
    report = {'archive': str(archive), 'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
              'extracted_directory': str(package), 'inputs_only_png': True,
              'input_sha256': input_hashes, 'fresh_inference': True, 'prediction_caches_used': False,
              'cwd_outside_package': True, 'original_project_data_in_archive': False,
              'native_smoke_test_passed': True, 'device': 'cpu', 'output': str(output),
              'both_log_locations_verified': True, 'adapter_matches_original_geometry_and_membership': True,
              'manifest_hashes_verified': True, 'adapter_unit_tests': 5, 'cases': counts,
              'case_seconds': {row['case_id']: row['seconds'] for row in decision['cases']},
              'wall_seconds': round(time.perf_counter()-tick, 3), 'docker_build_verified': False,
              'official_evaluation_run': False, 'official_score': None,
              'limitations': ['Docker CLI unavailable; Linux image build and GPU-container execution not tested',
                              'Official README and paper JSON schemas differ; sample layout is default',
                              'Existing full pipeline uses frozen v3 detector and U-Net v1; experimental v7 not integrated']}
    (ROOT/'dist/validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
