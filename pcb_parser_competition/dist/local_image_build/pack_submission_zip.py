"""Pack the saved image TAR into the upload ZIP as a single stored member.

.NET's ZipArchive maps CompressionLevel.NoCompression to DEFLATE level 0, which
still writes deflate framing (type 8) over an already-compressed `docker save`
payload and adds ~0.5 KB per MB. Python's zipfile emits a genuine STORED member
(type 0), so the ZIP is the TAR plus a header and a central directory, exactly as
the platform's "prebuilt image" path expects. The payload is streamed, never held
in memory, and ZIP64 stays available but is not forced.

Usage: pack_submission_zip.py <submission.tar> <upload.zip>
"""
import sys
import zipfile
from pathlib import Path

MEMBER = 'submission.tar'


def main():
    tar_path = Path(sys.argv[1])
    zip_path = Path(sys.argv[2])
    if not tar_path.is_file():
        raise SystemExit(f'missing image archive: {tar_path}')
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.write(tar_path, arcname=MEMBER)
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        assert names == [MEMBER], f'unexpected ZIP members: {names}'
        info = archive.getinfo(MEMBER)
        assert info.compress_type == zipfile.ZIP_STORED, 'member must be stored'
        assert info.file_size == tar_path.stat().st_size, 'member size differs from the TAR'
        assert info.compress_size == info.file_size, 'a stored member must not change size'
    print(f'{zip_path.name}: {zip_path.stat().st_size} bytes, single stored member {MEMBER}')


if __name__ == '__main__':
    main()