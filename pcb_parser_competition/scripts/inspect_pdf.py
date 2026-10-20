"""Minimal PDF inspector: is the document text or scanned images?

No third-party PDF library is available, so this walks the raw file, decompresses the
Flate streams and reports what it finds: how many pages, how many fonts, how many
images, and a sample of any text-show operators.  Enough to decide whether a pure
Python text extraction is worth attempting.
"""
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

PDF = Path(sys.argv[1] if len(sys.argv) > 1 else
           r"C:\Users\Pc\Desktop\新建文件夹\赛题六：AI驱动的PCB原理图自动解析（芯粒CAD和制造浙江省工程研究中心&比昂芯0803Rev1.pdf")


def main():
    raw = PDF.read_bytes()
    print(f"file: {PDF.name}")
    print(f"bytes: {len(raw)}")
    print(f"/Page objects : {len(re.findall(rb'/Type\s*/Page[^s]', raw))}")
    print(f"/Font refs    : {len(re.findall(rb'/Font', raw))}")
    print(f"/Image refs   : {len(re.findall(rb'/Image', raw))}")
    print(f"/DCTDecode    : {len(re.findall(rb'/DCTDecode', raw))}")
    print(f"/FlateDecode  : {len(re.findall(rb'/FlateDecode', raw))}")
    print(f"/ToUnicode    : {len(re.findall(rb'/ToUnicode', raw))}")

    streams = re.findall(rb'stream\r?\n(.*?)endstream', raw, re.S)
    print(f"streams: {len(streams)}")
    decoded = 0
    text_ops = Counter()
    samples = []
    for blob in streams:
        try:
            data = zlib.decompress(blob)
            decoded += 1
        except Exception:
            continue
        for op in (b'Tj', b'TJ', b'Tf', b'Do'):
            text_ops[op.decode()] += data.count(op)
        if b'Tj' in data or b'TJ' in data:
            for match in re.finditer(rb'\((?:[^()\\]|\\.)*\)\s*Tj', data):
                if len(samples) < 40:
                    samples.append(match.group()[:120])
    print(f"flate-decoded streams: {decoded}")
    print(f"operators: {dict(text_ops)}")
    print()
    print("sample text show operators:")
    for s in samples:
        print("  ", s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
