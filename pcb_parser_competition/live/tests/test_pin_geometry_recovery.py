"""Image and evidence boundary tests; no model weights or reference labels."""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, 'D:/pcb5_submission')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path('D:/pcb5_submission/src')))
import pcb_parser
pcb_parser.__path__.insert(0, str(ROOT/'src/pcb_parser'))
from pcb_parser.pin_geometry_recovery import DEFAULT, LeadEvidence, recover_image_leads, side_of, sync_networks, vote


def fixture():
    rgb = np.full((200, 200, 3), 255, np.uint8)
    cv2.rectangle(rgb, (50, 50), (130, 150), (150, 0, 0), 1)
    for y in (80, 120): cv2.line(rgb, (30, y), (50, y), (150, 0, 0), 1)
    target = {'components': {'X': {'type': 'ic', 'bbox': [50, 50, 130, 150]}},
              'pins': {'X': {'pin_1': {'pinname': 'IN', 'point': {'x': 30., 'y': 120.}}}}, 'nets': {}}
    records = [{'component_id': 'X', 'bbox_top_left': [50, 50, 130, 150]}]
    def reader(image, box, scale, rotation, recognition_only, view):
        return [{'text': t, 'confidence': .999, 'bbox': b, 'view': view} for t, b in
                [('1', [40, 70, 46, 76]), ('2', [40, 110, 46, 116]), ('OUT', [62, 115, 82, 125])]]
    return rgb, target, records, reader


class GeometryTests(unittest.TestCase):
    def recover(self, document=None, cfg=None, reader=None):
        rgb, target, records, default_reader = fixture()
        return recover_image_leads(target, document or {'pin_candidates': []}, {}, records, rgb,
                                   reader=reader or default_reader, cfg=cfg)

    def test_red_frame_and_leads(self):
        rgb, _, _, _ = fixture(); evidence = LeadEvidence(rgb, DEFAULT)
        frame = evidence.frame([50, 50, 130, 150])
        self.assertEqual(frame[0], [50., 50., 130., 150.])
        self.assertEqual([p['row'] for p in evidence.leads(frame[0], 'left')], [80, 120])
        self.assertEqual(evidence.leads(frame[0], 'right'), [])

    def test_grayscale_and_missing_border_abstain(self):
        rgb, _, _, _ = fixture(); rgb[:, :, 1:] = rgb[:, :, :1]
        self.assertIsNone(LeadEvidence(rgb, DEFAULT).frame([50, 50, 130, 150]))
        rgb, _, _, _ = fixture(); rgb[50, 50:131] = 255
        self.assertIsNone(LeadEvidence(rgb, DEFAULT).frame([50, 50, 130, 150]))

    def test_small_frame_abstains(self):
        rgb, _, _, _ = fixture()
        self.assertIsNone(LeadEvidence(rgb, DEFAULT).frame([50, 50, 70, 75]))

    def test_long_left_pin_near_corner_is_left(self):
        self.assertEqual(side_of([20, 53], [50, 50, 130, 150]), 'left')

    def test_requires_distinct_supported_views(self):
        token = {'text': '2', 'confidence': .99, 'view': 'one'}
        self.assertIsNone(vote([token, token], DEFAULT)[0])
        self.assertEqual(vote([token, {**token, 'view': 'two'}], DEFAULT)[0], '2')
        self.assertIsNone(vote([{**token, 'confidence': .97}, {**token, 'view': 'two'}], DEFAULT)[0])
        self.assertIsNone(vote([token, {**token, 'view': 'two'},
                               {**token, 'text': '3'}, {**token, 'text': '3', 'view': 'two'}], DEFAULT)[0])

    def test_recovers_only_visible_missing_row_without_mutation(self):
        rgb, target, records, reader = fixture(); before = copy.deepcopy(target)
        out, audit = recover_image_leads(target, {'pin_candidates': []}, {}, records, rgb, reader=reader)
        self.assertEqual(target, before)
        self.assertEqual(out['pins']['X']['pin_1'], before['pins']['X']['pin_1'])
        self.assertEqual(out['pins']['X']['pin_2']['pinname'], 'OUT')
        self.assertAlmostEqual(out['pins']['X']['pin_2']['point']['y'], 80)
        self.assertEqual(len(audit['changes']), 1)
        again, _ = recover_image_leads(out, {'pin_candidates': []}, {}, records, rgb, reader=reader)
        self.assertEqual(again, out)

    def test_one_view_and_empty_text_do_not_create_pin(self):
        _, _, _, reader = fixture()
        def repeated(*args): return [{**t, 'view': 'same'} for t in reader(*args)]
        self.assertEqual(self.recover(reader=repeated)[1]['changes'], [])
        self.assertEqual(self.recover(reader=lambda *args: [])[1]['changes'], [])

    def test_ambiguous_owner_can_be_resolved_but_multiple_candidates_abstain(self):
        candidate = {'candidate_id': 'C', 'point': [30, 120], 'owner_status': 'ambiguous_owner',
                     'owner_candidates': [{'component_id': 'X'}]}
        out, audit = self.recover({'pin_candidates': [candidate]})
        self.assertEqual(audit['changes'][0]['source_candidate_id'], 'C')
        self.assertEqual(out['pins']['X']['pin_2']['point']['x'], 30)
        self.assertEqual(self.recover({'pin_candidates': [candidate, {**candidate, 'candidate_id': 'D'}]})[1]['changes'], [])

    def test_ocr_budget_abstains(self):
        out, audit = self.recover(cfg={'max_ocr_calls': 1})
        self.assertEqual(audit['changes'], [])
        self.assertTrue(audit['budget_exhausted'])
        self.assertEqual(audit['ocr_calls'], 1)

    def test_ground_contact_needs_continuous_line_and_unique_anchor(self):
        rgb, target, _, _ = fixture()
        cv2.line(rgb, (90, 20), (90, 50), (150, 0, 0), 1)
        target['components']['G'] = {'type': 'gnd', 'bbox': [85, 175, 95, 185]}
        target['pins']['G'] = {'pin_1': {'pinname': 'GND', 'point': {'x': 90., 'y': 180.}}}
        evidence = LeadEvidence(rgb, DEFAULT)
        self.assertEqual(len(evidence.ground_anchors([90, 35], 'top', [50, 50, 130, 150], target)), 1)
        rgb[28:39, 88:93] = 255
        self.assertEqual(LeadEvidence(rgb, DEFAULT).ground_anchors([90, 35], 'top', [50, 50, 130, 150], target), [])


class NetworkTests(unittest.TestCase):
    def fixture(self):
        rgb = np.full((100, 100, 3), 255, np.uint8)
        cv2.line(rgb, (20, 20), (60, 20), (0, 150, 0), 1)
        out = {'nets': {'N': {'hyperGraph': '(Y.1)', 'edges': [[[20, 20], [60, 20]]]}}}
        graph = {'networks': [{'network_id': 'N', 'pin_candidate_ids': ['C'], 'edge_ids': ['E']}],
                 'edges': [{'edge_id': 'E', 'polyline': [[20, 20], [60, 20]]}],
                 'pin_attachments': [{'candidate_id': 'C', 'point': [20, 20],
                                      'attachment_status': 'attached', 'network_id': 'N'}]}
        change = {'owner': 'X', 'number': '2', 'name': 'OUT', 'status': 'image_lead_endpoint', 'point': [20, 20]}
        return rgb, out, graph, change

    def test_unique_existing_wire_attaches_without_changing_edges(self):
        rgb, out, graph, change = self.fixture(); edges = copy.deepcopy(out['nets']['N']['edges'])
        audit = sync_networks(out, graph, [change], LeadEvidence(rgb, DEFAULT))
        self.assertEqual(audit[0]['after'], ['X.2', 'Y.1'])
        self.assertEqual(out['nets']['N']['edges'], edges)
        self.assertEqual(sync_networks(out, graph, [change], LeadEvidence(rgb, DEFAULT)), [])

    def test_nc_and_missing_green_withhold(self):
        rgb, out, graph, change = self.fixture()
        self.assertNotIn('after', sync_networks(out, graph, [{**change, 'name': 'NC'}], LeadEvidence(rgb, DEFAULT))[0])
        rgb[:] = 255
        self.assertNotIn('after', sync_networks(out, graph, [change], LeadEvidence(rgb, DEFAULT))[0])

    def test_model_attachment_requires_exact_point_and_membership(self):
        rgb, out, graph, change = self.fixture(); change['source_candidate_id'] = 'C'
        graph['networks'][0]['pin_candidate_ids'] = []
        self.assertNotIn('after', sync_networks(out, graph, [change], LeadEvidence(rgb, DEFAULT))[0])
        graph['networks'][0]['pin_candidate_ids'] = ['C']; graph['pin_attachments'][0]['point'] = [21, 20]
        self.assertNotIn('after', sync_networks(out, graph, [change], LeadEvidence(rgb, DEFAULT))[0])
        graph['pin_attachments'][0]['point'] = [20, 20]
        self.assertIn('after', sync_networks(out, graph, [change], LeadEvidence(rgb, DEFAULT))[0])

    def test_missing_export_and_unresolved_junction_withhold(self):
        rgb, out, graph, change = self.fixture(); graph['networks'][0]['unresolved_junction_ids'] = ['J']
        self.assertNotIn('after', sync_networks(out, graph, [change], LeadEvidence(rgb, DEFAULT))[0])
        graph['networks'][0].pop('unresolved_junction_ids'); out['nets'] = {}
        self.assertNotIn('after', sync_networks(out, graph, [change], LeadEvidence(rgb, DEFAULT))[0])

    def test_disable_needs_no_artifacts_or_models(self):
        import agent
        target = {'anything': 'unchanged'}
        with patch.dict(os.environ, {'PCB_PIN_GEOMETRY_RECOVERY': '0'}):
            result, audit = agent.finalize_pin_geometry_recovery(target, Path('missing'), None, 'missing')
        self.assertIs(result, target)
        self.assertFalse(audit['enabled'])


if __name__ == '__main__': unittest.main()
