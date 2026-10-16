"""Check the 方式1 package tree against the frozen source, file by file.

Only two files may differ: the Dockerfile (BuildKit cache mounts removed) and
package_manifest.json (which records that Dockerfile hash). Everything that can
affect predictions — Python, weights, OCR models, configs, pin library — must be
byte-identical, so the platform-built image is the same algorithm as the verified
prebuilt one rather than a new variable.
"""
import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
FROZEN = PROJECT / 'submission' / 'platform_test_20261016_icled_gpu_1'
PACKAGE = Path(__file__).resolve().parent / 'package'
ALLOWED_DIFFERENCES = {'Dockerfile', 'package_manifest.json'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_tree(root):
    return {str(p.relative_to(root)).replace('\\', '/'): p for p in root.rglob('*') if p.is_file()}


def main():
    if not PACKAGE.is_dir():
        raise SystemExit(f'missing package directory: {PACKAGE}')
    frozen, package = relative_tree(FROZEN), relative_tree(PACKAGE)
    only_frozen = sorted(set(frozen) - set(package))
    only_package = sorted(set(package) - set(frozen))
    changed = sorted(name for name in set(frozen) & set(package) if digest(frozen[name]) != digest(package[name]))
    unexpected = sorted((set(only_frozen) | set(only_package) | set(changed)) - ALLOWED_DIFFERENCES)
    report = {
        'frozen_source_dir': str(FROZEN),
        'package_dir': str(PACKAGE),
        'frozen_file_count': len(frozen),
        'package_file_count': len(package),
        'missing_from_package': only_frozen,
        'extra_in_package': only_package,
        'changed_files': changed,
        'allowed_differences': sorted(ALLOWED_DIFFERENCES),
        'code_weights_and_configs_identical': not unexpected,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if unexpected:
        raise SystemExit(f'package diverges from the frozen source beyond the Dockerfile: {unexpected}')
    # The manifest must still describe the Dockerfile it now ships.
    manifest = json.loads((PACKAGE / 'package_manifest.json').read_text(encoding='utf-8'))
    entry = manifest['files']['Dockerfile']
    dockerfile = (PACKAGE / 'Dockerfile').read_bytes()
    if entry['sha256'] != digest(PACKAGE / 'Dockerfile') or entry['bytes'] != len(dockerfile):
        raise SystemExit('package_manifest.json does not match the shipped Dockerfile')
    for relative, expected in manifest['files'].items():
        if relative in ALLOWED_DIFFERENCES:
            continue
        data = (PACKAGE / relative).read_bytes()
        if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
            raise SystemExit(f'manifest hash mismatch in package: {relative}')
    for relative in manifest['runtime_assets']:
        if not (PACKAGE / relative).is_file():
            raise SystemExit(f'missing runtime asset: {relative}')
    print('manifest self-consistent: every shipped file matches package_manifest.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
