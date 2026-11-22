"""Strict validator mimicking PowerPoint's repair triggers."""
import io
import re
import sys
import zipfile
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
P = r'E:\AI_Workplace\精英挑战赛\EDA_Final_Defense\EDA精英挑战赛_决赛答辩_官方模板版.pptx'
z = zipfile.ZipFile(P)
names = z.namelist()

print('=== zip duplicate names ===')
dups = [n for n, c in Counter(names).items() if c > 1]
print(dups if dups else 'none')

print('\n=== presentation.xml sldIdLst ===')
pres = z.read('ppt/presentation.xml').decode('utf-8')
ids = re.findall(r'<p:sldId id="(\d+)" r:id="(rId\d+)"\s*/?>', pres)
print('entries:', len(ids))
idc = Counter(i for i, _ in ids)
dup_ids = [k for k, v in idc.items() if v > 1]
print('DUPLICATE sldId ids:', dup_ids if dup_ids else 'none')
ridc = Counter(r for _, r in ids)
dup_rids = [k for k, v in ridc.items() if v > 1]
print('DUPLICATE sldId r:ids:', dup_rids if dup_rids else 'none')

print('\n=== presentation rels ===')
rels = z.read('ppt/_rels/presentation.xml.rels').decode('utf-8')
relmap = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', rels))
missing_rel = [(i, r) for i, r in ids if r not in relmap]
print('sldId rIds missing in rels:', missing_rel if missing_rel else 'none')
# dangling rel targets
targets = re.findall(r'Target="([^"]+)"', rels)
import posixpath
dangling = []
for t in targets:
    if t.startswith('http') or t.startswith('../'):
        continue
    full = 'ppt/' + t
    if full not in names:
        dangling.append(t)
print('dangling rel targets:', dangling if dangling else 'none')
# parts on disk but not referenced
slide_parts = sorted(n for n in names if re.match(r'ppt/slides/slide\d+\.xml$', n))
referenced = {posixpath.normpath('ppt/' + relmap[r]) for _, r in ids if r in relmap}
unref = [s for s in slide_parts if s not in referenced]
print('slide parts on disk not in sldIdLst:', unref if unref else 'none')

print('\n=== content types coverage ===')
ct = z.read('[Content_Types].xml').decode('utf-8')
overrides = set(re.findall(r'PartName="([^"]+)"', ct))
check_parts = slide_parts + [n for n in names if re.match(r'ppt/slides/_rels/slide\d+\.xml\.rels$', n)]
missing_ct = [n for n in check_parts if ('/' + n) not in overrides]
print('parts missing CT override:', missing_ct if missing_ct else 'none')

print('\n=== per-slide checks ===')
for sp in slide_parts:
    xml = z.read(sp).decode('utf-8')
    relp = sp.replace('slides/', 'slides/_rels/') + '.rels'
    srels = z.read(relp).decode('utf-8') if relp in names else ''
    smap = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', srels))
    # layout rel present?
    has_layout = 'slideLayout' in srels
    # all r:embed/r:id referenced in xml resolve?
    used = set(re.findall(r'r:(?:embed|id|link)="(rId\d+)"', xml))
    unresolved = [r for r in used if r not in smap]
    # empty placeholders
    empty_ph = len(re.findall(r'<p:ph type="(?:title|ctrTitle|subTitle)"[^>]*/>', xml))
    # duplicate shape ids within slide
    sid = re.findall(r'<p:cNvPr id="(\d+)"', xml)
    dup_sid = [k for k, v in Counter(sid).items() if v > 1]
    texts = len(re.findall(r'<a:t>[^<]+</a:t>', xml))
    pics = xml.count('<p:pic>')
    print(f'{sp.split("/")[-1]:14s} texts={texts:3d} pics={pics} layoutRel={has_layout} '
          f'unresolvedRIds={unresolved if unresolved else "-"} emptyPh={empty_ph} dupShapeIds={dup_sid if dup_sid else "-"}')
