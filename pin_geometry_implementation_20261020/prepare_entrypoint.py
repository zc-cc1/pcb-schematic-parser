"""Extend the existing cached pipeline smoke check to both recovery hooks."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
s=(ROOT.parent/'pin_field_implementation_20261020/verify_entrypoint.py').read_text(encoding='utf-8')
s=s.replace('original_subprocess_run=subprocess.run',
            'original_geometry=agent.finalize_pin_geometry_recovery\noriginal_subprocess_run=subprocess.run')
s=s.replace("calls=[]", "calls=[]\ngeometry_calls=[]")
s=s.replace("source=ROOT/'ocr_cache'/(key+'.json')", "source=WORK/'pin_field_implementation_20261020/ocr_cache'/(key+'.json')")
anchor="with patch.object(agent.subprocess,'run'"
hook='''def geometry_reader(image,box,scale,rotation,recognition_only,view):
    crop=np.ascontiguousarray(image[box[1]:box[3],box[0]:box[2]])
    key=hashlib.sha256(crop.tobytes()+str((crop.shape,box,scale,rotation,recognition_only)).encode()).hexdigest()
    source=ROOT/'ocr_cache'/(key+'.json')
    assert source.exists(),f'Geometry OCR replay observation missing: {view}'
    return [{**t,'view':view} for t in read(source)]

def geometry_hook(target,stage,rgb,digest):
    before=copy.deepcopy(target)
    result,audit=original_geometry(target,stage,rgb,digest,reader=geometry_reader)
    assert target==before
    assert result['components']==before['components']
    for nid,net in before['nets'].items():assert result['nets'][nid]['edges']==net['edges']
    geometry_calls.append(audit)
    return result,audit

'''
s=s.replace(anchor,hook+anchor)
s=s.replace("side_effect=recovery_hook):", "side_effect=recovery_hook),patch.object(agent,'finalize_pin_geometry_recovery',side_effect=geometry_hook):")
s=s.replace("assert len(calls)==1", "assert len(calls)==1 and len(geometry_calls)==1\nassert 'pin_geometry_recovery_error' not in row,row\nassert row['pin_geometry_recovery']=={'enabled':True,'changed_pins':5,'changed_networks':0,'ocr_calls':14}\nassert (output/'logs/1_pin_geometry_recovery.json').is_file()")
s=s.replace("'field_log_written':True", "'geometry_log_written':True,'geometry_summary':row['pin_geometry_recovery'],'field_log_written':True")
(ROOT/'verify_entrypoint.py').write_text(s,encoding='utf-8')
print('Prepared entrypoint smoke check.')
