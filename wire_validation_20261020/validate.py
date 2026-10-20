"""Read-only scoring and isolated experiments; never edits the deployment tree."""
import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
DEPLOY = Path('D:/pcb5_submission')
TRUTH = WORKSPACE / '赛题六公开数据集/赛题六公开数据集/open_GT/open_GT'
sys.path.insert(0, str(WORKSPACE / 'pcb_parser_competition/scripts'))
import platform_metrics_v3 as metrics


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and p.suffix in {'.py', '.json', '.pt', '.onnx'}
            and '__pycache__' not in p.parts}


def agent_module(root):
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('wire_test_agent', root / 'agent.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(root):
    rows = []
    for folder in sorted(TRUTH.iterdir(), key=lambda p: int(p.name)):
        path = root / 'final_results' / f'{folder.name}_Pred.json'
        if not path.is_file():
            continue
        result = metrics.score_case(read(path), read(next(folder.glob('*.json'))))
        rows.append({'case_id': folder.name, **result})
    macro = {k: sum(r[k]['f1'] for r in rows) / len(rows)
             for k in metrics.WEIGHTS} if rows else {}
    return {'cases': len(rows), 'macro': macro,
            'score_100': sum(r['final_score'] for r in rows)/len(rows)*100 if rows else 0,
            'per_case': rows}


def protected_cleanup(agent):
    source = inspect.getsource(agent.cleanup_wire_edges)
    source = source.replace('min_length_px=8.0):', 'min_length_px=8.0, protection_radius=4.0, protected_points=None):')
    source = source.replace("for net in (target.get('nets') or {}).values():",
                            "for nid, net in (target.get('nets') or {}).items():")
    marker = '                    if angle(a, joined) > merge_degrees:'
    source = source.replace(marker,
        "                    if any(math.dist(shared, p) <= protection_radius for p in (protected_points or {}).get(nid, [])):\n"
        "                        continue\n" + marker)
    namespace = {}
    exec(compile(source, '<isolated-protected-cleanup>', 'exec'), namespace)
    return namespace['cleanup_wire_edges']


def points_by_net(payload, raw):
    out = {}
    for nid, net in payload['nets'].items():
        points = []
        for ref in str(net.get('hyperGraph', '')).strip('()').split(','):
            if '.' not in ref:
                continue
            cid, number = ref.strip().rsplit('.', 1)
            point = (raw.get('pins', {}).get(cid, {}).get('pin_' + number) or {}).get('point')
            if point:
                points.append([point['x'], point['y']])
        out[nid] = points
    return out


def replay(root, report_name):
    agent = agent_module(HERE / 'runtime')
    cleanup = protected_cleanup(agent)
    rows = []
    for folder in sorted(TRUTH.iterdir(), key=lambda p: int(p.name)):
        filename = f'{folder.name}_Pred.json'
        if not (root / 'final_results' / filename).is_file():
            continue
        pred, raw = read(root / 'final_results' / filename), read(root / 'component_results' / filename)
        truth = read(next(folder.glob('*.json')))
        before = copy.deepcopy(pred)
        for nid, net in before['nets'].items():
            net['edges'] = copy.deepcopy(raw['nets'][nid]['edges'])
        protection = points_by_net(before, raw)
        variants = {'existing_output': pred}
        variants['reproduced_cleanup'] = copy.deepcopy(before)
        agent.cleanup_wire_edges(variants['reproduced_cleanup'])
        for radius in [1.0, 4.0]:
            value = copy.deepcopy(before)
            cleanup(value, protection_radius=radius, protected_points=protection)
            variants[f'protect_{radius:g}px'] = value
        results = {}
        for name, value in variants.items():
            assert value['components'] == pred['components'] and value['pins'] == pred['pins']
            assert {k: n['hyperGraph'] for k,n in value['nets'].items()} == {k:n['hyperGraph'] for k,n in pred['nets'].items()}
            result = metrics.score_case(value, truth)
            results[name] = {**{k: result[k]['f1'] for k in metrics.WEIGHTS},
                             'score_100': result['final_score'] * 100,
                             'edge_count': sum(len(n['edges']) for n in value['nets'].values())}
        rows.append({'case': folder.name, 'metrics': results})
    names = rows[0]['metrics'] if rows else []
    macro = {n: {k: sum(r['metrics'][n][k] for r in rows)/len(rows)
                 for k in rows[0]['metrics'][n]} for n in names}
    result = {'source': str(root), 'cases': len(rows), 'macro': macro, 'per_case': rows,
              'note': 'Frozen existing predictions, only wire edges changed. No GT used by transformation.'}
    save(HERE / report_name, result)
    print(json.dumps({'cases':len(rows), 'macro':macro}, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['snapshot','verify','score','replay'])
    p.add_argument('--root', type=Path, default=HERE / 'baseline')
    p.add_argument('--report', default='replay.json')
    args = p.parse_args()
    if args.mode == 'snapshot':
        save(HERE / 'deployment_hashes_before.json', hashes(DEPLOY))
        print('Deployment snapshot saved.')
    elif args.mode == 'verify':
        before, after = read(HERE / 'deployment_hashes_before.json'), hashes(DEPLOY)
        changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
        save(HERE / 'deployment_integrity.json', {'unchanged': not changed, 'changed': changed, 'files':len(after)})
        print(json.dumps({'unchanged':not changed,'changed':changed,'files':len(after)}))
    elif args.mode == 'score':
        result=score(args.root);save(HERE / args.report,result)
        print(json.dumps({k:v for k,v in result.items() if k != 'per_case'},indent=2))
    else:
        replay(args.root, args.report)


if __name__ == '__main__':
    main()
