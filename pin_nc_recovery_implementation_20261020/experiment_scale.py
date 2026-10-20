"""Check the fixed 3x/4x/5x scale set on the complete numeral group."""
from pathlib import Path
exec(Path(__file__).with_name('experiment_binary.py').read_text(encoding='utf-8').split('result=[]')[0])
result=[]
for entry in read(ROOT/'experiment_grouped.json'):
    isolated=np.asarray(Image.open(ROOT/f"grouped_row_{entry['lead']['row']}.png").convert('RGB'))
    tokens=[]
    for scale in [3.,4.,5.]:tokens.extend(ocr(isolated,scale,f"group_gray_{entry['lead']['row']}_s{scale}"))
    value,support=vote([t for t in tokens if t['text'].isdigit()],DEFAULT)
    result.append({'row':entry['lead']['row'],'number':value,'support':support,'readings':tokens})
(ROOT/'experiment_scale.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
