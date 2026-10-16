from copy import deepcopy
import json
from pathlib import Path
import unittest

from pcb_parser.nearby_tip_fields import reconcile_nearby_tip_fields


def evidence(text, box):
    row = {'text': text, 'source_text': text, 'bbox': box, 'confidence': .999,
           'source_view': 'global', 'global_assignment_margin': 20, 'approximate_split_geometry': False}
    return {**row, 'alternatives': [{**row, 'source_view': 'tile'}]}


def fixture():
    record = {'detection_index': 0, 'component_id': 'U1', 'designator': 'U1', 'Name': 'IC',
              'type': 'ic', 'bbox_top_left': [40, 20, 80, 80]}
    common = {'owner': 'U1', 'owner_detection_index': 0, 'owner_status': 'assigned',
              'confidence': .9, 'text_side': 'left'}
    target = {**common, 'candidate_id': 'P1', 'point': [40, 50], 'number': '8', 'pinname': None,
              'number_status': 'observed_OCR_text', 'pinname_status': 'unresolved',
              'text_evidence': {'number': evidence('8', [30, 42, 39, 51])}}
    donor = {**common, 'candidate_id': 'P2', 'point': [43, 52], 'number': None, 'pinname': 'INT',
             'number_status': 'unresolved', 'pinname_status': 'observed_OCR_v2_missing_name_fill',
             'text_evidence': {'pinname': evidence('INT', [50, 42, 68, 57])}}
    pins = {'size': [100, 100], 'coordinate_origin': 'top_left', 'components': [deepcopy(record)], 'pin_candidates': [target, donor]}
    attachments = [{**{k: p[k] for k in ['candidate_id', 'point', 'owner', 'owner_status', 'number', 'pinname']},
                    'attachment_status': 'attached', 'network_id': 'L1', 'physical_network_id': 'N1',
                    'wire_projection': [p['point'][0], 50], 'wire_gap_px': 2, 'alternatives': []} for p in [target, donor]]
    net = {'network_id': 'L1', 'pin_candidate_ids': ['P1', 'P2'], 'edge_ids': ['W1'],
           'official_pin_references': ['U1.8'], 'unresolved_junction_ids': []}
    graph = {'pin_attachments': attachments, 'networks': [net], 'physical_networks': [{**deepcopy(net), 'network_id': 'N1'}],
             'edges': [{'edge_id': 'W1', 'polyline': [[10, 50], [44, 50]], 'mean_probability': .99}], 'gap_candidates': []}
    cfg = json.loads((Path(__file__).resolve().parents[1]/'configs/nearby_tip_fields_v1.json').read_text(encoding='utf-8'))
    return pins, graph, [record], cfg


class NearbyTipTests(unittest.TestCase):
    def run_fixture(self, payload):
        return reconcile_nearby_tip_fields(*payload)

    def withheld(self, payload, reason):
        new, graph, audit = self.run_fixture(payload)
        self.assertEqual(audit['summary']['names_reconciled'], 0)
        self.assertIsNone(new['pin_candidates'][0]['pinname'])
        self.assertIn(reason, audit['withheld'][0]['withheld_reasons'])

    def test_name_filled_without_numbers_points_members_or_donor_changes(self):
        payload = fixture(); old = deepcopy(payload)
        new, graph, audit = self.run_fixture(payload)
        self.assertEqual(audit['summary']['names_reconciled'], 1)
        self.assertEqual(new['pin_candidates'][0]['pinname'], 'INT')
        self.assertEqual(new['pin_candidates'][1], old[0]['pin_candidates'][1])
        self.assertEqual(graph['networks'], old[1]['networks'])
        self.assertEqual(graph['physical_networks'], old[1]['physical_networks'])
        self.assertEqual(graph['edges'], old[1]['edges'])
        self.assertEqual(payload, old)
        self.assertEqual(new['pin_candidates'][0]['text_evidence']['pinname']['observed_at_candidate_id'], 'P2')

    def test_reapplication_is_idempotent(self):
        p, g, r, c = fixture(); new, graph, _ = reconcile_nearby_tip_fields(p, g, r, c)
        again, ag, audit = reconcile_nearby_tip_fields(new, graph, r, c)
        self.assertEqual(again, new);self.assertEqual(ag, graph)
        self.assertEqual(audit['changes'], [])

    def test_existing_target_name_preserved(self):
        p, g, r, c = fixture();p['pin_candidates'][0]['pinname'] = 'ESTABLISHED';g['pin_attachments'][0]['pinname'] = 'ESTABLISHED'
        new, _, audit = reconcile_nearby_tip_fields(p, g, r, c)
        self.assertEqual(new['pin_candidates'][0]['pinname'], 'ESTABLISHED');self.assertEqual(audit['changes'], [])

    def test_number_is_not_transferred_to_donor(self):
        new, _, _ = self.run_fixture(fixture())
        self.assertIsNone(new['pin_candidates'][1]['number'])

    def test_distinct_physical_networks_reject_even_same_logical_net(self):
        f = fixture();f[1]['pin_attachments'][1]['physical_network_id'] = 'N2'
        self.withheld(f, 'different_or_unverified_physical_attachment')

    def test_membership_proof_required(self):
        f = fixture();f[1]['physical_networks'][0]['pin_candidate_ids'] = ['P1']
        self.withheld(f, 'different_or_unverified_physical_attachment')

    def test_ambiguous_attachment_rejected(self):
        f = fixture();f[1]['pin_attachments'][1]['attachment_status'] = 'ambiguous_nearby_networks'
        self.withheld(f, 'different_or_unverified_physical_attachment')

    def test_other_nearby_wire_candidate_rejected(self):
        f = fixture();f[1]['pin_attachments'][0]['alternatives'] = [{'physical_network_id': 'OTHER', 'distance_px': 2.5}]
        self.withheld(f, 'competing_wire_attachment')

    def test_unresolved_junction_rejected(self):
        f = fixture();f[1]['physical_networks'][0]['unresolved_junction_ids'] = ['J1']
        self.withheld(f, 'uncertain_network_topology')

    def test_suspected_gap_rejected(self):
        f = fixture();f[1]['gap_candidates'] = [{'network_ids': ['N1']}]
        self.withheld(f, 'uncertain_network_topology')

    def test_logical_power_merge_rejected(self):
        f = fixture();f[1]['networks'][0]['power_logical_merge'] = True
        self.withheld(f, 'uncertain_network_topology')

    def test_weak_edge_and_nonfinite_edge_rejected(self):
        for confidence in [.2, float('nan')]:
            with self.subTest(confidence=confidence):
                f = fixture();f[1]['edges'][0]['mean_probability'] = confidence
                self.withheld(f, 'no_unique_shared_physical_wire_segment')

    def test_separate_segments_on_same_network_rejected(self):
        f = fixture();f[1]['edges'][0]['polyline'] = [[10, 50], [40, 50]]
        self.withheld(f, 'no_unique_shared_physical_wire_segment')

    def test_overlapping_edges_do_not_prove_unique_lead(self):
        f = fixture();f[1]['edges'].append({**deepcopy(f[1]['edges'][0]), 'edge_id': 'W2'});f[1]['physical_networks'][0]['edge_ids'].append('W2')
        self.withheld(f, 'no_unique_shared_physical_wire_segment')

    def test_distant_projections_rejected(self):
        f = fixture();f[1]['pin_attachments'][1]['wire_projection'] = [47, 50]
        self.withheld(f, 'separated_wire_projections')

    def test_wire_gap_rejected(self):
        f = fixture();f[1]['pin_attachments'][0]['wire_gap_px'] = 4
        self.withheld(f, 'distant_wire_attachment')

    def test_different_side_rejected(self):
        f = fixture();f[0]['pin_candidates'][1]['text_side'] = 'top'
        self.withheld(f, 'different_or_unresolved_component_side')

    def test_missing_secondary_side_metadata_uses_geometry(self):
        f = fixture();f[0]['pin_candidates'][0]['text_side'] = None
        new, _, audit = self.run_fixture(f)
        self.assertEqual(new['pin_candidates'][0]['pinname'], 'INT')
        self.assertIsNone(new['pin_candidates'][0]['text_side'])
        self.assertEqual(audit['changes'][0]['component_side'], 'left')

    def test_corner_side_ambiguity_withheld(self):
        f = fixture()
        for pin, attachment, point in zip(f[0]['pin_candidates'], f[1]['pin_attachments'], [[41, 21], [42, 22]]):
            pin['point'] = point;attachment['point'] = point;pin['text_side'] = None
        self.withheld(f, 'different_or_unresolved_component_side')

    def test_wrong_row_glyph_rejected(self):
        f = fixture();e = f[0]['pin_candidates'][1]['text_evidence']['pinname'];e['bbox'] = [50, 60, 68, 74];e['alternatives'][0]['bbox'] = e['bbox']
        self.withheld(f, 'glyph_not_on_shared_lead_row')

    def test_cropped_or_single_view_name_rejected(self):
        for change in ['split', 'single']:
            with self.subTest(change=change):
                f = fixture();e = f[0]['pin_candidates'][1]['text_evidence']['pinname']
                if change == 'split':e['approximate_split_geometry'] = True
                else:e['alternatives'][0]['source_view'] = 'global'
                self.withheld(f, 'name_not_supported_by_repeated_literal_OCR')

    def test_derived_library_number_is_not_literal_number_evidence(self):
        f = fixture();f[0]['pin_candidates'][0]['number_status'] = 'library_inferred'
        self.withheld(f, 'number_not_supported_by_repeated_literal_OCR')

    def test_alternate_complete_spelling_blocks_fragment(self):
        f = fixture();e = f[0]['pin_candidates'][1]['text_evidence']['pinname'];e['alternatives'].append({**deepcopy(e['alternatives'][0]), 'text': 'INTERRUPT', 'source_text': 'INTERRUPT'})
        self.withheld(f, 'conflicting_observed_glyph_spelling')

    def test_third_model_tip_blocks_pair(self):
        f = fixture();p = deepcopy(f[0]['pin_candidates'][1]);p.update(candidate_id='P3', point=[42, 54], number=None, pinname=None)
        f[0]['pin_candidates'].append(p);a = deepcopy(f[1]['pin_attachments'][1]);a.update(candidate_id='P3', point=p['point'], number=None, pinname=None);f[1]['pin_attachments'].append(a)
        self.withheld(f, 'competing_nearby_tip')

    def test_duplicate_number_prevents_export_collision(self):
        f = fixture();p = deepcopy(f[0]['pin_candidates'][0]);p.update(candidate_id='P3', point=[40, 75]);f[0]['pin_candidates'].append(p)
        a = deepcopy(f[1]['pin_attachments'][0]);a.update(candidate_id='P3', point=p['point']);f[1]['pin_attachments'].append(a)
        self.withheld(f, 'number_already_claimed_by_another_tip')

    def test_weakened_config_rejected(self):
        f = fixture();f[3]['min_distinct_views'] = 1
        with self.assertRaises(ValueError):self.run_fixture(f)
        f = fixture();f[3]['max_tip_distance_px'] = 6
        with self.assertRaises(ValueError):self.run_fixture(f)


if __name__ == '__main__':
    unittest.main()
