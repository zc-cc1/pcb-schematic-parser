import hashlib,json,os,shutil
from pathlib import Path
def entry(image):
    index=json.loads(Path(os.environ['PCB_VALIDATION_CACHE_INDEX']).read_text(encoding='utf-8'))
    key=hashlib.sha256(Path(image).read_bytes()).hexdigest()
    return index[key]
def reuse_pin_folder(image, destination):
    source=Path(entry(image)['pin_folder'])
    assert json.loads((source/'pin_text_result.json').read_text(encoding='utf-8'))['image_sha256']==hashlib.sha256(Path(image).read_bytes()).hexdigest()
    shutil.copytree(source,destination)
    print('VALIDATION: reused identical frozen detection/OCR/pin predictions',flush=True)
def capacitor_inputs(image):
    value=entry(image)
    return [Path(p) for p in value['capacitor_probabilities']],Path(value['capacitor_metadata'])
