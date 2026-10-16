"""Package public symbol metadata, never per-image labels, for runtime use."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def prepare(source):
    original = json.loads(source.read_text(encoding='utf-8-sig'))
    parts = {}
    for model, record in original.items():
        by_number, by_name = defaultdict(set), defaultdict(set)
        for pin in record['pins']:
            number, name = str(pin.get('number', '')).strip(), str(pin.get('name', '')).strip()
            if not number or not name:
                continue
            by_number[number].add(name)
            by_name[name].add(number)
        parts[model] = {'number_to_names': {n: sorted(v) for n, v in sorted(by_number.items())},
                        'name_to_numbers': {n: sorted(v) for n, v in sorted(by_name.items())}}
    return {'schema_version': 1, 'source_kind': 'public_aggregate_symbol_library',
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'pin_array_order_has_no_geometry': True, 'parts': parts}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'models': len(result['parts']), 'source_sha256': result['source_sha256']}, ensure_ascii=False))
