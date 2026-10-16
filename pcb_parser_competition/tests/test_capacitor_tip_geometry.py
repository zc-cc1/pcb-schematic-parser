from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

from pcb_parser.capacitor_tip_geometry import refine_capacitor_geometry, supported_extension
from pcb_parser.engineering_network_export import wire_axis
from test_engineering_network_export import fixture as base_fixture, config as engineering_config


def config():return json.loads((Path(__file__).resolve().parents[1]/'configs/capacitor_tip_geometry_v1.json').read_text(encoding='utf-8'))


def heat(points, confidence):
    yy, xx = np.mgrid[:100, :100];a = np.zeros((100, 100), np.float32)
    for x, y in points:a = np.maximum(a, confidence*np.exp(-((xx-x)**2+(yy-y)**2)/2))
    return a.astype(np.float32)


def fixture():
    p, g, r, ec = base_fixture('capacitor', [40, 30, 80, 70])
    for i, point in enumerate([[40, 50], [80, 50]]):
        p['pin_candidates'][i]['point'] = point;g['pin_attachments'][i].update(point=point, wire_projection=point)
    g['edges'][0].update(polyline=[[10, 50], [40, 50]], length_px=30)
    g['edges'][1].update(polyline=[[80, 50], [95, 50]], length_px=15)
    g['nodes'] = [{'node_id': n, 'point': point} for n, point in [('A', [10, 50]), ('B', [40, 50]), ('C', [80, 50]), ('D', [95, 50])]]
    maps = []
    for left, right, confidence in [(40, 80, .99), (44, 76, .92), (48, 72, .8)]:
        wire = np.zeros((100, 100), np.float32);wire[50, 10:left+1] = .99;wire[50, right:96] = .99
        maps.append(np.stack([wire, heat([[left, 50], [right, 50]], confidence)]))
    return p, g, r, *maps, config(), ec


class CapacitorGeometryTests(unittest.TestCase):
    def test_coordinate_is_exact_zoom_peak_with_supported_local_extension(self):
        f = fixture();old = deepcopy(f);p, g, a = refine_capacitor_geometry(*f)
        self.assertEqual([x['point'] for x in p['pin_candidates']], [[48, 50], [72, 50]])
        self.assertEqual(a['summary']['refined_tips'], 2);self.assertEqual(f[:3], old[:3])
        self.assertEqual(g['edges'][0]['polyline'], [[10, 50], [40, 50], [48, 50]])
        self.assertEqual(g['physical_networks'], old[1]['physical_networks'])
        self.assertEqual(g['networks'], old[1]['networks'])
        self.assertEqual(p['pin_candidates'][0]['native_model_point'], [40, 50])
        self.assertAlmostEqual(g['pin_attachments'][0]['wire_gap_px'], 0)

    def test_existing_fields_protected_and_both_leads_commit_together(self):
        f = fixture();f[0]['pin_candidates'][0]['number'] = '1'
        p, g, a = refine_capacitor_geometry(*f)
        self.assertEqual(a['changes'], []);self.assertEqual(p, f[0]);self.assertEqual(g, f[1])

    def test_missing_low_confidence_zoom_peak_blocks_both_leads(self):
        f = fixture();f[5][1] = heat([[48, 50]], .8)
        self.assertEqual(refine_capacitor_geometry(*f)[2]['summary']['refined_tips'], 0)

    def test_wire_gap_is_not_invented_or_filled(self):
        f = fixture();f[5][0, 50, 45] = 0
        p, g, a = refine_capacitor_geometry(*f)
        self.assertEqual(a['changes'], []);self.assertEqual(g['edges'], f[1]['edges'])

    def test_outward_or_nonmonotonic_peak_does_not_move_tip(self):
        f = fixture();f[5][1] = heat([[42, 50], [78, 50]], .8)
        self.assertEqual(refine_capacitor_geometry(*f)[2]['summary']['refined_tips'], 0)

    def test_two_leads_in_same_physical_net_are_not_refined(self):
        f = fixture();f[1]['pin_attachments'][1]['physical_network_id'] = 'N1'
        f[1]['physical_networks'][0]['edge_ids'].append('E2')
        self.assertEqual(refine_capacitor_geometry(*f)[2]['summary']['refined_tips'], 0)

    def test_competing_local_peaks_require_review(self):
        f = fixture();f[5][1] = heat([[48, 50], [44, 50], [72, 50]], .8)
        self.assertEqual(refine_capacitor_geometry(*f)[2]['summary']['refined_tips'], 0)

    def test_tangential_shift_cannot_jump_to_neighbor_row(self):
        f = fixture();f[5][1] = heat([[48, 54], [72, 54]], .8)
        self.assertEqual(refine_capacitor_geometry(*f)[2]['summary']['refined_tips'], 0)

    def test_extension_near_other_wire_does_not_change_graph(self):
        f = fixture();f[1]['edges'].append({'edge_id': 'OTHER', 'source': 'O1', 'target': 'O2', 'polyline': [[46, 45], [46, 55]], 'mean_probability': .99})
        f[1]['nodes'] += [{'node_id': 'O1', 'point': [46, 45]}, {'node_id': 'O2', 'point': [46, 55]}]
        self.assertEqual(refine_capacitor_geometry(*f)[2]['summary']['refined_tips'], 0)

    def test_other_component_blocks_extension(self):
        f = fixture();other = {'component_id': 'X1', 'detection_index': 1, 'type': 'ic', 'bbox_top_left': [44, 48, 47, 52], 'designator': 'X1'}
        f[2].append(other);f[0]['components'].append(deepcopy(other))
        self.assertEqual(refine_capacitor_geometry(*f)[2]['summary']['refined_tips'], 0)

    def test_reversed_polyline_uses_geometric_node_endpoint(self):
        f = fixture();f[1]['edges'][0]['polyline'].reverse()
        _, g, a = refine_capacitor_geometry(*f)
        self.assertEqual(a['summary']['refined_tips'], 2);self.assertEqual(g['edges'][0]['polyline'], [[48, 50], [40, 50], [10, 50]])

    def test_short_extension_keeps_pixel_bend_and_local_lead_axis(self):
        f = fixture();f[1]['edges'][0]['polyline'][-1] = [40, 51];f[1]['nodes'][1]['point'] = [40, 51]
        f[1]['pin_attachments'][0]['wire_projection'] = [40, 51];f[3][0, 51, 40] = .99
        f[4][1] = heat([[42, 50], [78, 50]], .92);f[5][1] = heat([[44, 50], [76, 50]], .8)
        p, g, a = refine_capacitor_geometry(*f)
        self.assertEqual(a['summary']['refined_tips'], 2)
        self.assertEqual(wire_axis(p['pin_candidates'][0], g, engineering_config())[0], 0)

    def test_scalar_score_and_raw_probability_sources_are_explicit(self):
        p, _, _ = refine_capacitor_geometry(*fixture());pin = p['pin_candidates'][0]
        self.assertAlmostEqual(pin['confidence'], (.99+.92+.8)/3, places=5)
        self.assertIn('not_calibrated', pin['confidence_aggregation']);self.assertIsNone(pin['number']);self.assertIsNone(pin['pinname'])

    def test_invalid_probability_dimensions_nan_and_bounds_rejected(self):
        for value in [np.zeros((2, 2, 2)), np.full((2, 100, 100), np.nan), np.full((2, 100, 100), 1.1)]:
            f = list(fixture());f[3] = value
            with self.assertRaises(ValueError):refine_capacitor_geometry(*f)

    def test_unqualified_type_and_unknown_identity_unchanged(self):
        for field, value in [('type', 'resistor'), ('designator', None)]:
            f = fixture();f[2][0][field] = value;f[0]['components'][0][field] = value
            self.assertEqual(refine_capacitor_geometry(*f)[2]['summary']['refined_tips'], 0)


if __name__ == '__main__':unittest.main()
