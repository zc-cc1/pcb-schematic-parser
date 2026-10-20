"""Fresh detection OCR on a single lead row preserves glyph context."""
from pathlib import Path
exec(Path(__file__).with_name('experiment_binary.py').read_text(encoding='utf-8').split('result=[]')[0])
result=[]
for entry in read(ROOT/'experiment_grouped.json'):
    row=entry['lead']['row'];edge=entry['lead']['edge'];box=[math.floor(edge-40),math.floor(row-17),math.ceil(edge+65),math.ceil(row+19)]
    if engine is None:engine,_=create_engine({'onnx_threads':4})
    tokens=[]
    for scale in [3.,5.]:
        crop=np.ascontiguousarray(rgb[box[1]:box[3],box[0]:box[2]])
        key=hashlib.sha256(crop.tobytes()+str((crop.shape,box,scale,0,False)).encode()).hexdigest();dest=ROOT/'ocr_cache'/(key+'.json')
        if dest.exists():value=read(dest)
        else:
            value=recognize(engine,rgb,box,scale,0,False,f'nc_row_{row}_s{scale}')
            dest.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
        tokens.extend(value)
    result.append({'row':row,'box':box,'tokens':tokens})
(ROOT/'experiment_row.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
