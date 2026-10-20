"""Correct exported pin positions using image evidence from the same body side.

No pin is added, removed, renumbered or renamed. Reference labels are never read.
"""
import math
import statistics
from collections import defaultdict


IC_TYPES = {'ic', 'box', 'connector', 'raw_block', 'block'}


def correct_pin_positions_by_side(target, original_pins, image_rgb, measure_lead):
    """Refine legacy tip snapping in place; return a per-call evidence audit.

    ``original_pins`` is the snapshot BEFORE legacy snapping. Inward centroid
    estimates can therefore vote, and a prior bad snap cannot create consensus.
    Coordinates in target/original_pins are bottom-left; image is top-left.
    """
    height, width = image_rgb.shape[:2]
    audit = {'policy': 'same_side_image_length_consensus_v1', 'enabled': True,
             'modified_pin_count': 0, 'groups': []}
    for component, pins in original_pins.items():
        comp = target.get('components', {}).get(component, {})
        output_pins = target.get('pins', {}).get(component, {})
        box = comp.get('bbox') or []
        if comp.get('type') not in IC_TYPES or len(box) != 4:
            continue
        x1, y1, x2, y2 = map(float, box)
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x1 >= x2 or y1 >= y2:
            continue
        groups = defaultdict(list)
        for key, pin in pins.items():
            if key not in output_pins:
                continue
            point = pin.get('point') or {}
            if 'x' not in point or 'y' not in point:
                continue
            x, y = float(point['x']), float(point['y'])
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            distances = {'left': abs(x-x1), 'right': abs(x-x2),
                         'top': abs(y-y2), 'bottom': abs(y-y1)}
            side = min(distances, key=distances.get)
            if distances[side] > 60:
                continue
            length = measure_lead(image_rgb, width, height,
                                  (x1, height-y2, x2, height-y1), side, (x, height-y))
            groups[side].append({'key': key, 'raw_point': {'x': x, 'y': y}, 'length': length})
        for side, rows in groups.items():
            valid = [r for r in rows if r['length'] is not None
                     and math.isfinite(r['length']) and 1 <= r['length'] <= 60]
            clusters = []
            for row in valid:
                cluster = [r for r in valid if abs(r['length']-row['length']) <= 3]
                median = statistics.median(r['length'] for r in cluster)
                # Deterministic ties: compact cluster first, then shorter length.
                rank = (len(cluster), -sum(abs(r['length']-median) for r in cluster), -median)
                clusters.append((rank, cluster))
            cluster = max(clusters, key=lambda item: item[0])[1] if clusters else []
            axis = 'y' if side in {'left', 'right'} else 'x'
            distinct = []
            for value in sorted(r['raw_point'][axis] for r in cluster):
                if not distinct or value-distinct[-1] > 6:
                    distinct.append(value)
            accepted = len(distinct) >= 3 and len(cluster) > len(rows)/2
            length = statistics.median(r['length'] for r in cluster) if accepted else None
            entry = {'component': component, 'side': side, 'measurements': rows,
                     'accepted': accepted, 'consensus_length': length, 'changes': []}
            if accepted:
                for row in rows:
                    point = output_pins[row['key']]['point']
                    new = dict(point)
                    if side == 'left':
                        new['x'] = x1-length
                    elif side == 'right':
                        new['x'] = x2+length
                    elif side == 'top':
                        new['y'] = y2+length
                    else:
                        new['y'] = y1-length
                    delta = math.hypot(point['x']-new['x'], point['y']-new['y'])
                    if 0 < delta <= 25:
                        entry['changes'].append({'key': row['key'], 'before': dict(point), 'after': new})
                        point.update(new)
                        audit['modified_pin_count'] += 1
            audit['groups'].append(entry)
    return audit
