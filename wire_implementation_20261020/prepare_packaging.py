"""Stage packaging metadata without changing the deployed project."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
DEPLOY=Path('D:/pcb5_submission')
RUNTIME=ROOT/'runtime'
for name in ['Dockerfile.overlay','overlay_check.py','package_manifest.json']:
    shutil.copy2(DEPLOY/name,ROOT/'backup'/name)
    shutil.copy2(DEPLOY/name,RUNTIME/name)
docker=(RUNTIME/'Dockerfile.overlay').read_text(encoding='utf-8')
anchor='COPY agent.py /submission/agent.py\n'
assert docker.count(anchor)==1
docker=docker.replace(anchor,anchor+
    'COPY src/pcb_parser/wire_graph.py /submission/src/pcb_parser/wire_graph.py\n'
    'COPY src/pcb_parser/power_wire_export.py /submission/src/pcb_parser/power_wire_export.py\n')
(RUNTIME/'Dockerfile.overlay').write_text(docker,encoding='utf-8')
overlay=(RUNTIME/'overlay_check.py').read_text(encoding='utf-8')
overlay=overlay.replace('ROOT = pathlib.Path("/submission")',
                        'ROOT = pathlib.Path(__file__).resolve().parent')
anchor='    "src/pcb_parser/export_geometry.py",\n'
assert overlay.count(anchor)==1
overlay=overlay.replace(anchor,anchor+'    "src/pcb_parser/wire_graph.py",\n    "src/pcb_parser/power_wire_export.py",\n')
overlay=overlay.replace('from pcb_parser import export_geometry as geom',
    'from pcb_parser import export_geometry as geom\n'
    'from pcb_parser.power_wire_export import restore_power_wires\n'
    'assert callable(restore_power_wires), "power wire exporter is unavailable"')
(RUNTIME/'overlay_check.py').write_text(overlay,encoding='utf-8')
for folder, pattern in [('src','*.py'),('scripts','*.py'),('configs','*.json')]:
    for source in (DEPLOY/folder).rglob(pattern):
        destination=RUNTIME/source.relative_to(DEPLOY)
        if not destination.exists():
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination)
for relative in ['run.sh','component_export.py']:
    shutil.copy2(DEPLOY/relative,RUNTIME/relative)
print('Staged Docker overlay and module import check.')
