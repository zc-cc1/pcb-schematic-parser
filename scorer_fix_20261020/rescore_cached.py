"""Re-evaluate existing artifacts only. Does not run or import inference code."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'pcb_parser_competition/scripts'
sys.path.insert(0, str(SCRIPTS))
import platform_metrics_v3 as fixed
from platform_wire_metrics import WIRE_SCORER_VERSION


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy_base = load('legacy_base', ROOT / 'scorer_fix_20261020/original_platform_metrics.py')
legacy = load('legacy_v3', ROOT / 'scorer_fix_20261020/original_platform_metrics_v3.py')
legacy.old = legacy_base
sources = {
    'original_baseline': ROOT / 'wire_validation_20261020/baseline/final_results',
    'path_preserving': ROOT / 'wire_validation_small_20261020/baseline',
    'restore_physical_contacts': ROOT / 'wire_validation_small_20261020/restore_physical_contacts',
}
rows = []
for case in map(str, range(1, 14)):
    truth = read(next((ROOT / '赛题六公开数据集/赛题六公开数据集/open_GT/open_GT' / case).glob('*.json')))
    row = {'case': case, 'variants': {}}
    for variant, folder in sources.items():
        prediction = read(folder / f'{case}_Pred.json')
        before, after = legacy.score_case(prediction, truth), fixed.score_case(prediction, truth)
        unchanged = all(before[k] == after[k] for k in fixed.WEIGHTS if k != 'wire_f1')
        assert unchanged, (case, variant)
        row['variants'][variant] = {'legacy_wire': before['wire_f1'],
                                    'fixed_wire': after['wire_f1'],
                                    'non_wire_metrics_unchanged': unchanged}
    rows.append(row)

groups = {'initial_three': {'5', '6', '13'},
          'remaining_ten': set(map(str, range(1, 14))) - {'5', '6', '13'},
          'all_thirteen': set(map(str, range(1, 14)))}
summary = {}
for group, cases in groups.items():
    selected = [row for row in rows if row['case'] in cases]
    summary[group] = {'cases': len(selected), 'variants': {}}
    for variant in sources:
        summary[group]['variants'][variant] = {
            metric: sum(row['variants'][variant][metric]['f1'] for row in selected) / len(selected)
            for metric in ['legacy_wire', 'fixed_wire']}
    changes = [row['variants']['restore_physical_contacts']['fixed_wire']['f1'] -
               row['variants']['path_preserving']['fixed_wire']['f1'] for row in selected]
    summary[group]['power_restore_improved_equal_worse'] = [sum(d > 0 for d in changes),
                                                           sum(d == 0 for d in changes),
                                                           sum(d < 0 for d in changes)]
result = {'wire_scorer_version': WIRE_SCORER_VERSION, 'neural_inference_calls': 0,
          'summary': summary, 'per_case': rows,
          'note': 'Macro wire F1 only; other local metrics unchanged and aggregate remains a proxy.'}
(ROOT / 'scorer_fix_20261020/cached_rescore.json').write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
