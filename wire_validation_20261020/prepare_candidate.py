"""Build a disposable runtime with path-preserving splitting and frozen upstream reuse."""
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
BASE=HERE/'runtime'
DEST=HERE/'runtime_candidate'

SPLIT=r'''    # Validation variant: cut on the actual raster path, preserving all bends.
    pin_splits = 0
    pin_split_limit = float(cfg.get('pin_split_max_distance_px') or 0)
    if pin_split_limit > 0:
        pin_points = np.asarray([p['point'] for p in pin_result['pin_candidates']], dtype=float).reshape(-1, 2)
        next_node = 0
        for u, v, key, data in list(graph.edges(keys=True, data=True)):
            path = np.asarray(data['path'], dtype=float)
            if len(path) < 3 or not len(pin_points):
                continue
            arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            if arc[-1] < 2 * pin_split_limit:
                continue
            cuts = set()
            for point in pin_points:
                distances = np.linalg.norm(path - point, axis=1)
                index = int(distances.argmin())
                if (distances[index] <= pin_split_limit
                        and pin_split_limit < arc[index] < arc[-1] - pin_split_limit):
                    cuts.add(index)
            if not cuts:
                continue
            # NetworkX's edge ordering may differ from stored path ordering.
            if math.dist(graph.nodes[u]['point'], path[0]) > math.dist(graph.nodes[v]['point'], path[0]):
                u, v = v, u
            graph.remove_edge(u, v, key)
            boundaries = [0] + sorted(cuts) + [len(path) - 1]
            previous = u
            for part, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
                fragment = path[begin:end+1].tolist()
                if part == len(boundaries) - 2:
                    target = v
                else:
                    target = f'pin_split_{next_node:05d}'
                    next_node += 1
                    graph.add_node(target, kind='pin_split', point=fragment[-1], degree=2)
                edge_id = f"{data['edge_id']}_{part+1}"
                graph.add_edge(previous, target, key=edge_id, edge_id=edge_id,
                               kind=data.get('kind', 'wire'), path=fragment,
                               polyline=polyline(fragment, cfg['polyline_tolerance_px']),
                               length_px=float(arc[end] - arc[begin]),
                               mean_probability=data.get('mean_probability', 0.0))
                previous = target
            pin_splits += len(cuts)
'''


def prepare():
    if not DEST.exists():
        shutil.copytree(BASE, DEST, ignore=shutil.ignore_patterns('__pycache__'))
    source=(BASE/'src/pcb_parser/wire_graph.py').read_text(encoding='utf-8')
    begin=source.index('    pin_splits=0\n')
    end=source.index('    nets,attachments=network_membership(',begin)
    (DEST/'src/pcb_parser/wire_graph.py').write_text(source[:begin]+SPLIT+source[end:],encoding='utf-8')
    script=(BASE/'scripts/run_component_pin_library.py').read_text(encoding='utf-8')
    original="        subprocess.run([sys.executable, str(ROOT/'scripts/run_pin_text_v2.py'), '--image', str(a.image),\n                        '--output', str(folder), '--device', a.device], check=True)"
    replacement="        from validation_cache import reuse_pin_folder\n        reuse_pin_folder(a.image, folder)"
    assert script.count(original)==1
    (DEST/'scripts/run_component_pin_library.py').write_text(script.replace(original,replacement),encoding='utf-8')
    script=(BASE/'scripts/run_capacitor_tip_geometry.py').read_text(encoding='utf-8')
    original="    result = process(a.image, stage, a.output, read(a.config), device=a.device)"
    replacement="    from validation_cache import capacitor_inputs\n    paths, meta = capacitor_inputs(a.image)\n    result = process(a.image, stage, a.output, read(a.config), probability_paths=paths, cache_metadata=meta, device=a.device)"
    assert script.count(original)==1
    (DEST/'scripts/run_capacitor_tip_geometry.py').write_text(script.replace(original,replacement),encoding='utf-8')
    helper='''import hashlib,json,os,shutil
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
'''
    (DEST/'scripts/validation_cache.py').write_text(helper,encoding='utf-8')
    print('Prepared isolated candidate runtime:',DEST)


def index():
    entries={}
    for work in (HERE/'baseline/work').iterdir():
        pin_folder=work/'p_stage23_stage21_stage20_stage18/stage17/stage16/upstream/upstream'
        capacitor=work/'p_stage23'
        if not (capacitor/'probability_inputs.json').is_file():
            continue
        pin=json.loads((pin_folder/'pin_text_result.json').read_text(encoding='utf-8'))
        paths=sorted((capacitor/'probabilities').glob('scale_*.npz'))
        assert len(paths)==3
        entries[pin['image_sha256']]={'pin_folder':str(pin_folder),
            'capacitor_probabilities':[str(p) for p in paths],
            'capacitor_metadata':str(capacitor/'probability_inputs.json')}
    (HERE/'cache_index.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
    print('Frozen caches:',len(entries))


if __name__=='__main__':
    import sys
    index() if len(sys.argv)>1 and sys.argv[1]=='index' else prepare()
