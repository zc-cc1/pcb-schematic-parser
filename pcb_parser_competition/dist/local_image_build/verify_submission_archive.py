"""Verify a platform upload archive: ZIP layout plus the TAR it contains.

Checks are deliberately redundant with `docker load`: they must hold even if the
daemon is unavailable, and they pin the archive to an expected image identity.

Two details of `docker save` output this handles:
  * BuildKit writes a docker-compatible `manifest.json` next to content-addressed
    `blobs/sha256/<digest>` members. The blob directory also holds the image
    config, the image manifest and a provenance/attestation manifest, so the blob
    count deliberately exceeds the layer count.
  * `Layers` may repeat a digest: an empty `{}` layer is shared by several build
    stages, so 17 layer entries can map to 14 unique blobs.

Usage: verify_submission_archive.py <submission.tar> <upload.zip> [--expect-image-id sha256:...]
Prints a JSON report on stdout and exits non-zero on the first failed check.
"""
import hashlib
import json
import sys
import tarfile
import zipfile
from pathlib import Path

CHUNK = 8 << 20


def sha256_stream(stream):
    digest = hashlib.sha256()
    while True:
        block = stream.read(CHUNK)
        if not block:
            break
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path):
    with open(path, 'rb') as stream:
        return sha256_stream(stream)


def verify_zip(zip_path, tar_path=None):
    """Validate layout and CRC. With tar_path, also pin the member bytes."""
    result = {}
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        result['zip_entries'] = names
        assert names == ['submission.tar'], f'unexpected ZIP members: {names}'
        info = archive.getinfo('submission.tar')
        result['zip_entry_bytes'] = info.file_size
        if tar_path:
            assert info.file_size == Path(tar_path).stat().st_size, 'ZIP member size differs from the TAR on disk'
        assert info.compress_type == zipfile.ZIP_STORED, 'member must be stored, not re-compressed'
        # testzip() streams every member and validates each CRC-32.
        assert archive.testzip() is None, 'ZIP CRC check failed'
        with archive.open('submission.tar') as member:
            result['zip_member_sha256'] = sha256_stream(member)
        if tar_path:
            assert result['zip_member_sha256'] == sha256_file(tar_path), 'ZIP member differs from the TAR on disk'
    result['zip_bytes'] = Path(zip_path).stat().st_size
    result['zip_sha256'] = sha256_file(zip_path)
    result['zip_crc_verified'] = True
    result['zip_layout'] = 'single stored member submission.tar'
    return result


def verify_tar(tar_path, expect_image_id=None):
    result = {'tar_bytes': Path(tar_path).stat().st_size}
    result['tar_sha256'] = sha256_file(tar_path)
    with tarfile.open(tar_path, 'r:') as archive:
        members = {member.name: member for member in archive.getmembers()}
        result['tar_members'] = len(members)

        # Every content-addressed member must hash to the digest it is named by.
        blobs = 0
        blob_bytes = 0
        for name, member in members.items():
            if not name.startswith('blobs/sha256/'):
                continue
            with archive.extractfile(member) as stream:
                assert sha256_stream(stream) == name.rsplit('/', 1)[-1], f'blob digest mismatch: {name}'
            blobs += 1
            blob_bytes += member.size
        # Legacy docker-format layers are stored as <digest>/layer.tar.
        legacy = 0
        for name, member in members.items():
            if not name.endswith('/layer.tar'):
                continue
            with archive.extractfile(member) as stream:
                assert sha256_stream(stream) == name.split('/')[0], f'layer digest mismatch: {name}'
            legacy += 1
        assert blobs or legacy, 'no layer blobs found in the archive'
        result['content_addressed_blobs_verified'] = blobs
        result['legacy_layer_tars_verified'] = legacy
        result['blob_payload_bytes'] = blob_bytes
        result['tar_format'] = 'docker save (manifest.json' + (' + OCI blobs)' if blobs else ')')
        result['referenced_only_as_metadata'] = (
            'blobs/ also holds the image config, image manifest and provenance manifest members, '
            'so the blob count exceeds the layer count by design')

        assert 'manifest.json' in members, 'docker-compatible manifest.json is missing'
        manifest = json.load(archive.extractfile('manifest.json'))
        assert len(manifest) == 1, 'archive must describe exactly one image'
        entry = manifest[0]
        result['image_tags'] = entry.get('RepoTags')

        config_bytes = archive.extractfile(entry['Config']).read()
        config_hex = entry['Config'].rsplit('/', 1)[-1].replace('.json', '')
        assert hashlib.sha256(config_bytes).hexdigest() == config_hex, 'config digest mismatch'
        result['config_digest'] = 'sha256:' + config_hex
        result['image_id'] = result['config_digest']

        # `docker image inspect {{.Id}}` reports the digest of the manifest list
        # BuildKit exported, which is the descriptor in `index.json` — not the
        # per-platform config digest inside `manifest.json`. Pin against the
        # descriptor when the archive has an OCI index, else against the config.
        indexes = []
        if 'index.json' in members:
            indexes = [descriptor['digest'] for descriptor in json.load(archive.extractfile('index.json')).get('manifests', [])]
        result['manifest_list_digests'] = indexes
        identity = indexes[0] if indexes else result['config_digest']
        result['identity_matched_against'] = 'index.json descriptor' if indexes else 'config digest'
        if expect_image_id:
            assert identity == expect_image_id, (
                f'archive identity is {identity} ({result["identity_matched_against"]}), expected {expect_image_id}')

        # os/architecture sit at the top level of the image config; Entrypoint and
        # the environment live in its nested runtime `config` object.
        image_config = json.loads(config_bytes)
        runtime_config = image_config.get('config') or {}
        result['os'] = image_config.get('os')
        result['architecture'] = image_config.get('architecture')
        result['entrypoint'] = runtime_config.get('Entrypoint')
        result['image_env'] = [item for item in runtime_config.get('Env') or [] if item.startswith('PCB_')]
        assert image_config.get('os') == 'linux', 'image must be a linux image'
        assert image_config.get('architecture') == 'amd64', 'image must be amd64'
        assert runtime_config.get('Entrypoint') == ['/submission/run.sh'], 'entrypoint must be /submission/run.sh'
        assert 'PCB_DEVICE=auto' in (runtime_config.get('Env') or []), 'deployment must default to PCB_DEVICE=auto'
        assert 'PCB_OUTPUT_FORMAT=target' in (runtime_config.get('Env') or []), 'deployment must default to the Target protocol'

        # Layer accounting: the manifest lists one entry per filesystem layer,
        # matching the config's diff_ids, but empty layers repeat one digest.
        layers = entry.get('Layers', [])
        missing = [name for name in layers if name not in members]
        assert not missing, f'layer blobs referenced but absent: {missing}'
        diff_ids = image_config.get('rootfs', {}).get('diff_ids', [])
        assert len(layers) == len(diff_ids), f'{len(layers)} layer entries vs {len(diff_ids)} config diff_ids'
        result['layer_entries'] = len(layers)
        result['unique_layer_blobs'] = len(set(layers))
        result['config_diff_ids'] = len(diff_ids)
        result['layer_blobs_verified'] = len(layers)

        # The optional OCI index must point at a member of this archive.
        if 'index.json' in members:
            index = json.load(archive.extractfile('index.json'))
            for descriptor in index.get('manifests', []):
                name = 'blobs/sha256/' + descriptor['digest'].split(':', 1)[1]
                assert name in members, f'index.json references a missing blob: {name}'
            result['oci_index_manifests'] = len(index.get('manifests', []))
    return result


def main():
    tar_path, zip_path = sys.argv[1], sys.argv[2]
    expect = None
    if '--expect-image-id' in sys.argv:
        expect = sys.argv[sys.argv.index('--expect-image-id') + 1]
    report = {'tar': verify_tar(tar_path, expect), 'zip': verify_zip(zip_path, tar_path)}
    report['all_checks_passed'] = True
    text = json.dumps(report, indent=2)
    # --out writes UTF-8 from Python itself; a PowerShell `1>` redirect would
    # produce UTF-16, which is unreadable to most downstream tooling.
    if '--out' in sys.argv:
        Path(sys.argv[sys.argv.index('--out') + 1]).write_text(text, encoding='utf-8')
    else:
        print(text)
    print(f"archive verified: {report['zip']['zip_bytes']:,} byte ZIP, single stored member, "
          f"{report['tar']['layer_entries']} layers, identity {report['tar']['identity_matched_against']}")
    print(text)


if __name__ == '__main__':
    main()