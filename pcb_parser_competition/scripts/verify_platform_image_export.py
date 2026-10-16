"""Verify the uploaded ZIP against its successfully loaded Docker save archive."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile


def digest(stream):
    result = hashlib.sha256()
    count = 0
    while chunk := stream.read(8 * 1024 * 1024):
        result.update(chunk)
        count += len(chunk)
    return result.hexdigest(), count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('build_root', type=Path)
    args = parser.parse_args()
    root = args.build_root.resolve()
    report_path = root / 'build_report.json'
    report = json.loads(report_path.read_text(encoding='utf-8-sig'))
    archive = Path(report['upload_zip'])
    original_tar = root / 'submission.tar'
    assert archive.parent == root and archive.name in {'submission_target.zip', 'submission_official_slim.zip'}
    assert report['docker_archive_load_passed'] and report['target_layout_verified']
    assert report['source_manifest_matches_image'] and report['all_bundled_file_hashes_verified']
    with archive.open('rb') as stream:
        zip_sha, zip_bytes = digest(stream)
    assert zip_sha == report['upload_zip_sha256'].lower()
    assert zip_bytes == report['upload_zip_bytes']
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == ['submission.tar']
        # Reading through EOF also verifies this member's ZIP CRC.
        with bundle.open('submission.tar') as stream:
            payload_sha, payload_bytes = digest(stream)
    with original_tar.open('rb') as stream:
        tar_sha, tar_bytes = digest(stream)
    assert payload_sha == tar_sha and payload_bytes == tar_bytes
    with tarfile.open(original_tar, mode='r:') as image:
        manifest = json.load(image.extractfile('manifest.json'))
        assert len(manifest) == 1 and manifest[0]['RepoTags'] == [report['image']]
        config_name = manifest[0]['Config']
        config_bytes = image.extractfile(config_name).read()
        assert hashlib.sha256(config_bytes).hexdigest() == config_name.rsplit('/', 1)[-1].removesuffix('.json')
        config = json.loads(config_bytes)
        assert (config['os'], config['architecture']) == ('linux', 'amd64')
        assert config['config']['Entrypoint'] == ['/submission/run.sh']
        assert 'PCB_OUTPUT_FORMAT=target' in config['config']['Env']
    verification = {
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'zip': str(archive), 'zip_bytes': zip_bytes, 'zip_sha256': zip_sha,
        'zip_single_member': 'submission.tar', 'zip_crc_verified': True,
        'payload_matches_loaded_tar': True, 'tar_sha256': tar_sha,
        'tar_bytes': tar_bytes, 'single_image_verified': True,
        'image': report['image'], 'platform': 'linux/amd64',
        'entrypoint': ['/submission/run.sh'], 'output_format': 'target',
        'official_score': None,
    }
    integrity_path = root / 'export_integrity.json'
    integrity_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    report['export_integrity'] = verification
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
