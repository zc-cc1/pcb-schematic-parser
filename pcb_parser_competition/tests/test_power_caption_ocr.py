"""Localized glyph geometry, ambiguity, and integration with power gates."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'tests'))
from test_power_networks import fixture, CFG as POWER, GUARD
from pcb_parser.power_networks import propose_power_networks
from pcb_parser.power_caption_ocr import recovery_rois, recover_power_captions

read = lambda n: json.loads((ROOT/'configs'/n).read_text(encoding='utf-8'))
CFG = read('power_caption_ocr_v1.json')
LABELS = read('network_labels_v1.json')


class FakeOCR:
    def __init__(self, rois, cfg, text='GND', boundary=False, weak=False, conflict=False, empty=False):
        self.rois, self.cfg, self.text = rois, cfg, text
        self.boundary, self.weak, self.conflict, self.empty = boundary, weak, conflict, empty
        self.calls = 0

    def __call__(self, pixels, **options):
        assert options == {'use_det': True, 'use_cls': True, 'use_rec': True}
        views = len(self.cfg['scales'])*len(self.cfg['rotations'])
        roi = self.rois[self.calls//views]
        rotation = self.cfg['rotations'][(self.calls % views)//len(self.cfg['scales'])]
        scale = self.cfg['scales'][self.calls % len(self.cfg['scales'])]
        self.calls += 1
        if self.empty:
            return SimpleNamespace(boxes=None, txts=None, scores=None)
        x1, y1, x2, y2 = roi['crop']
        x = 30+roi['detection_index']*130
        points = np.array([[x-12, 60], [x+12, 60], [x+12, 72], [x-12, 72]], float)-[x1, y1]
        if self.boundary:
            points[:, 0] = [0, 24, 24, 0]
        if rotation == 90:
            points = np.column_stack(((y2-y1-1)-points[:, 1], points[:, 0]))
        points *= scale
        text = 'DGND' if self.conflict and self.calls % views < 3 else self.text
        return SimpleNamespace(boxes=[points], txts=[text], scores=[.90 if self.weak else .999])


class PowerCaptionOCRTests(unittest.TestCase):
    def setup_inputs(self, **options):
        p, g, r, t = fixture()
        # Original caption has one observed view and cannot pass S15.
        t = t[::3]
        _, audit = propose_power_networks(p, g, r, t, POWER, GUARD)
        cfg = deepcopy(CFG)
        cfg.update(options)
        rois, _ = recovery_rois(p, g, r, audit, cfg)
        return p, g, r, t, audit, cfg, rois

    def recover(self, inputs, **options):
        p, g, r, t, audit, cfg, rois = inputs
        engine = FakeOCR(rois, cfg, **options)
        out, texts, recovery = recover_power_captions(np.zeros((140, 360, 3), np.uint8),
            p, g, r, t, audit, engine, cfg, POWER, LABELS)
        return out, texts, recovery, engine

    def test_detected_boxes_restore_to_original_pixels_for_all_scales_and_rotations(self):
        inputs = self.setup_inputs()
        p, g, r, t, audit, cfg, rois = inputs
        snapshot = deepcopy(inputs)
        out, texts, recovery, engine = self.recover(inputs)
        self.assertEqual(engine.calls, 12)
        for text in texts[len(t):]:
            x = 30+text['target_detection_index']*130
            np.testing.assert_allclose(text['bbox'], [x-12, 60, x+12, 72])
        self.assertEqual(out['edges'], g['edges'])
        self.assertEqual(out['networks'], g['networks'])
        self.assertEqual(out['pin_attachments'], g['pin_attachments'])
        self.assertEqual(inputs, snapshot)
        result, after = propose_power_networks(p, out, r, texts, POWER, GUARD)
        self.assertEqual(after['summary']['power_groups_applied'], 1)

    def test_no_original_bound_sites_can_be_recovered_from_actual_localized_text(self):
        inputs = self.setup_inputs()
        p, g, r, t, audit, cfg, rois = inputs
        g['label_sites'] = []
        out, texts, recovery, _ = self.recover(inputs)
        self.assertEqual(recovery['new_power_caption_sites'], 2)
        _, after = propose_power_networks(p, out, r, texts, POWER, GUARD)
        self.assertEqual(after['summary']['observed_power_sites'], 2)

    def test_empty_OCR_changes_nothing(self):
        inputs = self.setup_inputs()
        out, texts, recovery, _ = self.recover(inputs, empty=True)
        self.assertEqual(out, inputs[1])
        self.assertEqual(texts, inputs[3])
        self.assertEqual(recovery['localized_observations'], 0)

    def test_boundary_glyphs_do_not_become_votes_or_caption_sites(self):
        inputs = self.setup_inputs()
        out, texts, recovery, _ = self.recover(inputs, boundary=True)
        self.assertEqual(out, inputs[1])
        self.assertEqual(texts, inputs[3])
        self.assertEqual(len(recovery['rejected_boundary_observations']), 12)

    def test_weak_views_do_not_relax_existing_power_confidence_gate(self):
        inputs = self.setup_inputs()
        out, texts, _, _ = self.recover(inputs, weak=True)
        _, after = propose_power_networks(inputs[0], out, inputs[2], texts, POWER, GUARD)
        self.assertEqual(after['summary']['power_groups_applied'], 0)

    def test_conflicting_names_are_retained_and_withheld(self):
        inputs = self.setup_inputs()
        out, texts, _, _ = self.recover(inputs, conflict=True)
        self.assertEqual({t['text'] for t in texts}, {'GND', 'DGND'})
        _, after = propose_power_networks(inputs[0], out, inputs[2], texts, POWER, GUARD)
        self.assertEqual(after['summary']['power_groups_applied'], 0)

    def test_component_field_overlap_prevents_new_caption_assignment(self):
        inputs = self.setup_inputs()
        inputs[1]['label_sites'] = []
        for r in inputs[2]:
            x = 30+r['detection_index']*130
            r['fields'] = {'Name': {'bbox': [x-12, 60, x+12, 72]}}
        out, _, recovery, _ = self.recover(inputs)
        self.assertEqual(recovery['new_power_caption_sites'], 0)

    def test_close_competing_symbols_on_same_net_do_not_claim_one_caption(self):
        inputs = self.setup_inputs()
        p, g, r = inputs[:3]
        component = {**p['components'][0], 'detection_index': 2, 'component_id': 'OTHER'}
        p['components'].append(component)
        pin = {**p['pin_candidates'][0], 'candidate_id': 'P2', 'owner_detection_index': 2, 'owner': 'OTHER'}
        p['pin_candidates'].append(pin)
        g['pin_attachments'].append({**pin, 'attachment_status': 'attached', 'network_id': 'N0'})
        out, _, recovery, _ = self.recover(inputs)
        self.assertTrue(all(s.get('detection_index') != 0 for s in out['label_sites'][len(g['label_sites']):]))

    def test_budget_limits_OCR_work_and_reports_skips(self):
        inputs = self.setup_inputs(max_rois=1)
        out, _, recovery, engine = self.recover(inputs)
        self.assertEqual(recovery['roi_count'], 1)
        self.assertEqual(recovery['skipped_by_budget'], 1)
        self.assertEqual(engine.calls, 6)

    def test_missing_or_ambiguous_attachment_is_not_a_recovery_target(self):
        inputs = self.setup_inputs()
        inputs[1]['pin_attachments'][0]['attachment_status'] = 'ambiguous'
        rois, _ = recovery_rois(inputs[0], inputs[1], inputs[2], inputs[4], inputs[5])
        self.assertEqual([r['detection_index'] for r in rois], [1])

    def test_duplicate_scale_views_are_invalid(self):
        inputs = self.setup_inputs(scales=[2, 2, 3])
        with self.assertRaises(ValueError):
            self.recover(inputs)

    def test_wrong_RGB_size_is_rejected(self):
        p, g, r, t, audit, cfg, rois = self.setup_inputs()
        with self.assertRaises(ValueError):
            recover_power_captions(np.zeros((141, 360, 3), np.uint8), p, g, r, t, audit,
                                   FakeOCR(rois, cfg), cfg, POWER, LABELS)


if __name__ == '__main__':
    unittest.main()
