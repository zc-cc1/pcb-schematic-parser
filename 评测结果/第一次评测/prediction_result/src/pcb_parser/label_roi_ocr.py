"""Local RGB recognizer evidence only; no spelling replacement or net merging."""
from collections import Counter
import math
import time

import cv2
import numpy as np


def collect_label_readings(rgb, sites, engine, cfg):
    height, width = rgb.shape[:2]
    rows = []
    chosen = [s for s in sites if s.get('status') == 'bound_ordinary_label'][:cfg['max_sites']]
    for site in chosen:
        bb = site['bbox']
        crop = [max(0, math.floor(bb[0])-cfg['padding_x_px']), max(0, math.floor(bb[1])-cfg['padding_y_px']),
                min(width, math.ceil(bb[2])+cfg['padding_x_px']), min(height, math.ceil(bb[3])+cfg['padding_y_px'])]
        x1, y1, x2, y2 = crop
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Invalid label crop')
        patch = rgb[y1:y2, x1:x2]
        observations = []
        for scale in cfg['scales']:
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError('Invalid OCR scale')
            scaled = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            begin = time.perf_counter()
            result = engine(np.ascontiguousarray(scaled[:, :, ::-1]), use_det=False, use_cls=False, use_rec=True)
            for text, score in zip(result.txts if result.txts is not None else [], result.scores if result.scores is not None else []):
                score = float(score)
                if not math.isfinite(score):
                    raise ValueError('Nonfinite OCR confidence')
                observations.append({'text': str(text), 'confidence': score, 'scale': scale,
                                     'crop_top_left': crop, 'view': f'label_rec_s{scale}',
                                     'geometry_scope': 'whole_crop_recognition_not_word_localization',
                                     'wall_seconds': time.perf_counter()-begin})
        votes = Counter(o['text'] for o in observations if o['confidence'] >= cfg['min_suggestion_confidence'])
        unanimous = next((text for text, count in votes.items() if count == len(cfg['scales']) and count == len(observations)), None)
        rows.append({'site_id': site['site_id'], 'original_label': site['label'], 'crop_top_left': crop,
                     'observations': observations, 'high_confidence_scale_votes': dict(votes),
                     'suggested_spelling': unanimous if unanimous and unanimous != site['label'] else None,
                     'applied': False, 'crop_may_contain_adjacent_row': True})
    return {'readings': rows, 'roi_count': len(rows), 'suggestion_count': sum(r['suggested_spelling'] is not None for r in rows),
            'spelling_changes_applied': 0, 'RGB_only': True,
            'scope': 'Review evidence from the same OCR model at several scales, not independent votes or verified correctness'}
