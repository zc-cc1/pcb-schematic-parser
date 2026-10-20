"""Stage a narrow, hash-guarded update of the live D: project."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
TARGET=Path('D:/pcb5_submission')
sha=lambda data:hashlib.sha256(data).hexdigest()
path=TARGET/'agent.py'
original=path.read_bytes()
source=original.decode('utf-8').replace('\r\n','\n')
old='def pin_lead_snap(target, lead_px, image_rgb=None, mode="color"):'
new='def pin_lead_snap(target, lead_px, image_rgb=None, mode="color", side_audit=None):'
assert source.count(old)==1
source=source.replace(old,new)
old='    moved = measured = unified = 0\n    components = target.get(\'components\') or {}'
new='''    # Keep a pre-snap snapshot so corrected coordinates cannot vote for themselves.
    side_enabled = (image_rgb is not None and mode == "color" and
                    os.environ.get('PCB_PIN_SIDE_CONSENSUS', '1').strip().lower()
                    not in {'0', 'off', 'false'})
    original_pins = None
    if side_enabled:
        from copy import deepcopy
        original_pins = deepcopy(target.get('pins') or {})
    if side_audit is not None:
        side_audit.clear()
        side_audit.update(enabled=False, modified_pin_count=0, groups=[])
    moved = measured = unified = 0
    components = target.get('components') or {}'''
assert source.count(old)==1
source=source.replace(old,new)
old='    pin_lead_snap.measured = measured\n    pin_lead_snap.unified = unified\n    return moved'
new='''    if side_enabled:
        from pcb_parser.pin_position_consensus import correct_pin_positions_by_side
        correction = correct_pin_positions_by_side(target, original_pins, image_rgb, measure_lead)
        if side_audit is not None:
            side_audit.update(correction)
    pin_lead_snap.measured = measured
    pin_lead_snap.unified = unified
    return moved'''
assert source.count(old)==1
source=source.replace(old,new)
old='                _moved = pin_lead_snap(target, float(_lead), _rgb, _tip_mode)'
new='''                _side_audit = {}
                _moved = pin_lead_snap(target, float(_lead), _rgb, _tip_mode, side_audit=_side_audit)
                row['pin_side_consensus'] = {key: _side_audit[key] for key in
                                              ['enabled', 'modified_pin_count']}
                write(output/'logs'/f'{case_id}_pin_side_consensus.json', _side_audit)'''
assert source.count(old)==1
source=source.replace(old,new)
staged=ROOT/'runtime/agent.py'
staged.write_text(source,encoding='utf-8',newline='\n')

paths=['agent.py','src/pcb_parser/pin_position_consensus.py',
       'tests/test_pin_position_consensus.py','docs/pin_position_consensus.md']
files=[]
for name in paths:
    target=TARGET/name
    data=(ROOT/'runtime'/name).read_bytes()
    previous=target.read_bytes() if target.exists() else None
    if previous is not None:
        backup=ROOT/'backup'/name
        backup.parent.mkdir(parents=True,exist_ok=True)
        if backup.exists():assert backup.read_bytes()==previous
        else:backup.write_bytes(previous)
    files.append({'path':name,'sha256':sha(data),'previous_sha256':sha(previous) if previous is not None else None})
(ROOT/'release_plan.json').write_text(json.dumps({'target':str(TARGET),'files':files},indent=2)+'\n',encoding='utf-8')
print('Staged',len(files),'files; original agent snapshot saved.')
