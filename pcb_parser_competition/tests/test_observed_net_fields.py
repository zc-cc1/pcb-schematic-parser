from copy import deepcopy
import json
from pathlib import Path
import unittest

from pcb_parser.observed_net_fields import complete_observed_net_fields, correct_body_half
from verify_observed_net_fields_runtime import geometry_bijection


ROOT = Path(__file__).resolve().parents[1]
def config(name):return json.loads((ROOT/'configs'/name).read_text(encoding='utf-8'))


def reading(text, box, views=3):
    return {'text': text, 'bbox': box, 'confidence': .999, 'view': 'v0',
            'observations': [{'text': text, 'bbox': box, 'confidence': .999, 'view': f'v{i}'} for i in range(views)]}


def fixture(unresolved=False):
    identity = 'UNRESOLVED_DET_0001' if unresolved else 'U1'
    r = {'detection_index': 0, 'component_id': identity, 'designator': None if unresolved else identity,
        'type': 'ic', 'bbox_top_left': [60, 50, 200, 150], 'detection_confidence': .99, 'Name': None, 'value': None, 'fields': {}}
    positions = [[60, 70], [60, 100], [60, 130], [200, 70], [200, 130]] if unresolved else [[60, 100], [200, 100]]
    pins = {'size': [300, 250], 'components': [deepcopy(r)], 'pin_candidates': []}
    names = ['VIN', 'GND', 'EN', 'VOUT', 'BYP']
    for i, point in enumerate(positions):
        name = names[i] if unresolved else None
        pins['pin_candidates'].append({'candidate_id': f'P{i}', 'point': point, 'confidence': .99,
            'owner': identity, 'owner_detection_index': 0, 'owner_status': 'assigned',
            'number': str(i+1), 'number_status': 'observed_OCR_text', 'pinname': name,
            'pinname_status': 'observed_OCR_text' if name else 'ambiguous_text',
            'text_evidence': {'number': {'text': str(i+1)}, 'pinname': {'text': name}} if name else {'number': {'text': str(i+1)}},
            'text_candidates': {'number': [], 'pinname': []}})
    graph = {'edges': [], 'nodes': [], 'physical_networks': [], 'gap_candidates': [],
        'networks': [{'network_id': 'N', 'pin_candidate_ids': [p['candidate_id'] for p in pins['pin_candidates']], 'official_pin_references': []}],
        'pin_attachments': [deepcopy(p) for p in pins['pin_candidates']]}
    texts = [reading('L1', [62, 42, 79, 54]), reading('LDO', [62, 148, 90, 160])] if unresolved else [
        reading('CAGC', [72, 94, 115, 104]), reading('DOUT', [145, 94, 188, 104])]
    return pins, graph, texts, [r], config('observed_net_fields_v1.json'), config('pin_text_v2.json')


class ObservedSideNameTests(unittest.TestCase):
    def test_two_sides_same_row_get_their_own_complete_glyphs(self):
        p, g, r, a = complete_observed_net_fields(*fixture())
        self.assertEqual([x['pinname'] for x in p['pin_candidates']], ['CAGC', 'DOUT'])
        self.assertEqual(a['summary']['observed_pin_names_added'], 2)
        self.assertTrue(p['pin_candidates'][0]['text_evidence']['pinname']['observed_OCR_glyph'])

    def test_crossing_midpoint_is_withheld_even_with_many_views(self):
        f = fixture();f[2][0] = reading('CAGC', [72, 94, 135, 104], 5)
        self.assertIsNone(complete_observed_net_fields(*f)[0]['pin_candidates'][0]['pinname'])

    def test_one_view_cannot_fill(self):
        f = fixture();f[2][0] = reading('CAGC', [72, 94, 115, 104], 1)
        self.assertIsNone(complete_observed_net_fields(*f)[0]['pin_candidates'][0]['pinname'])

    def test_low_confidence_or_conflicting_spelling_is_withheld(self):
        for conflicting in [False, True]:
            f = fixture()
            if conflicting:f[2].append(reading('CAGG', [72, 94, 115, 104], 2))
            else:
                f[2][0]['confidence'] = .97
                for obs in f[2][0]['observations']:obs['confidence'] = .97
            self.assertIsNone(complete_observed_net_fields(*f)[0]['pin_candidates'][0]['pinname'])

    def test_number_missing_and_secondary_tip_stay_unfilled(self):
        for key, value in [('number', None), ('text_hypothesis_primary_candidate_id', 'OTHER')]:
            f = fixture();f[0]['pin_candidates'][0][key] = value
            self.assertIsNone(complete_observed_net_fields(*f)[0]['pin_candidates'][0]['pinname'])

    def test_existing_name_and_its_glyph_are_protected(self):
        f = fixture();pin = f[0]['pin_candidates'][1]
        pin.update(pinname='EXISTING', text_evidence={'pinname': {'bbox': [72, 94, 115, 104]}})
        p, _, _, _ = complete_observed_net_fields(*f)
        self.assertIsNone(p['pin_candidates'][0]['pinname']);self.assertEqual(p['pin_candidates'][1]['pinname'], 'EXISTING')

    def test_component_value_box_is_not_reused_as_name(self):
        f = fixture();f[3][0]['fields']['Name'] = {'bbox': [72, 94, 115, 104]}
        self.assertIsNone(complete_observed_net_fields(*f)[0]['pin_candidates'][0]['pinname'])

    def test_neighboring_row_competition_is_not_silently_resolved(self):
        f = fixture();p = deepcopy(f[0]['pin_candidates'][0]);p.update(candidate_id='P2', point=[60, 102])
        f[0]['pin_candidates'].append(p);f[1]['pin_attachments'].append(deepcopy(p));f[1]['networks'][0]['pin_candidate_ids'].append('P2')
        result = complete_observed_net_fields(*f)[0]
        self.assertIsNone(result['pin_candidates'][0]['pinname']);self.assertIsNone(result['pin_candidates'][2]['pinname'])

    def test_top_bottom_and_entire_body_containment(self):
        ctx = {'component': {'bbox_top_left': [0, 0, 100, 100]}, 'side': 'top'}
        self.assertTrue(correct_body_half({'bbox': [40, 10, 60, 30]}, ctx, 1))
        self.assertFalse(correct_body_half({'bbox': [40, 40, 60, 55]}, ctx, 1))
        ctx['side'] = 'bottom';self.assertTrue(correct_body_half({'bbox': [40, 60, 60, 80]}, ctx, 1))
        self.assertFalse(correct_body_half({'bbox': [40, 60, 110, 80]}, ctx, 1))


class ObservedHeaderTests(unittest.TestCase):
    def test_visible_L_prefix_is_kept_for_IC_and_references_are_synchronized(self):
        f = fixture(True);original = deepcopy(f);p, g, r, a = complete_observed_net_fields(*f)
        self.assertEqual(r[0]['designator'], 'L1');self.assertEqual(r[0]['Name'], 'LDO');self.assertEqual(r[0]['type'], 'ic')
        self.assertEqual([x['owner'] for x in p['pin_candidates']], ['L1']*5)
        self.assertEqual(g['networks'][0]['official_pin_references'], [f'L1.{i}' for i in range(1, 6)])
        self.assertEqual(a['summary']['new_pin_numbers'], 0);self.assertEqual(f, original)
        self.assertFalse(r[0]['fields']['designator']['canonical_CAD_identity_inferred'])

    def test_footer_missing_or_ambiguous_is_withheld(self):
        for ambiguous in [False, True]:
            f = fixture(True)
            if ambiguous:f[2].append(reading('LDR', [62, 148, 90, 160]))
            else:f[2].pop()
            self.assertEqual(complete_observed_net_fields(*f)[2][0]['component_id'], 'UNRESOLVED_DET_0001')

    def test_single_title_view_is_withheld(self):
        f = fixture(True);f[2][0] = reading('L1', [62, 42, 79, 54], 1)
        self.assertIsNone(complete_observed_net_fields(*f)[2][0]['designator'])

    def test_header_in_pin_row_is_not_a_title(self):
        f = fixture(True);f[2][0] = reading('L1', [72, 91, 89, 104])
        self.assertIsNone(complete_observed_net_fields(*f)[2][0]['designator'])

    def test_repeated_title_at_another_location_is_not_unique(self):
        for views in [1, 3]:
            f = fixture(True);f[2].append(reading('L1', [240, 42, 257, 54], views))
            self.assertIsNone(complete_observed_net_fields(*f)[2][0]['designator'])

    def test_full_title_and_footer_boxes_must_clear_pin_rows(self):
        for idx, text, box in [(0, 'L1', [62, 42, 79, 69]), (1, 'LDO', [62, 132, 90, 160])]:
            f = fixture(True);f[2][idx] = reading(text, box)
            self.assertIsNone(complete_observed_net_fields(*f)[2][0]['designator'])

    def test_occupied_title_and_nonliteral_substring_are_withheld(self):
        for occupied in [False, True]:
            f = fixture(True)
            if occupied:f[3][0]['fields']['value'] = {'bbox': [62, 42, 79, 54]}
            else:f[2][0] = reading('L1 U2', [62, 42, 100, 54])
            self.assertIsNone(complete_observed_net_fields(*f)[2][0]['designator'])

    def test_unobserved_pin_numbers_or_names_cannot_validate_header(self):
        for field in ['number', 'pinname']:
            f = fixture(True)
            for p in f[0]['pin_candidates']:p[field+'_status'] = 'engineering_convention_unverified'
            self.assertIsNone(complete_observed_net_fields(*f)[2][0]['designator'])

    def test_low_detection_and_duplicate_are_withheld(self):
        for key, value in [('detection_confidence', .7), ('duplicate_of_detection_index', 3)]:
            f = fixture(True);f[3][0][key] = value
            self.assertIsNone(complete_observed_net_fields(*f)[2][0]['designator'])

    def test_existing_identity_and_existing_model_name_are_protected(self):
        f = fixture(True);f[3][0]['Name'] = 'OLD_MODEL'
        self.assertEqual(complete_observed_net_fields(*f)[2][0]['Name'], 'OLD_MODEL')
        f = fixture(True);f[3][0]['designator'] = 'U1'
        self.assertEqual(complete_observed_net_fields(*f)[2][0]['designator'], 'U1')

    def test_existing_ID_collision_and_unsupported_profile_are_withheld(self):
        f = fixture(True);r = deepcopy(f[3][0]);r.update(component_id='L1', designator='L1', detection_index=1)
        f[3].append(r);f[0]['components'].append(deepcopy(r))
        self.assertIsNone(complete_observed_net_fields(*f)[2][0]['designator'])
        f = fixture();f[4]['official_identity_verified'] = True
        with self.assertRaises(ValueError):complete_observed_net_fields(*f)


class RuntimeContactTests(unittest.TestCase):
    def test_confidence_order_swaps_are_compared_by_unique_position(self):
        a = [{'candidate_id': 'P1', 'point': [10, 10]}, {'candidate_id': 'P2', 'point': [20, 20]}]
        b = [{'candidate_id': 'P1', 'point': [20.001, 20]}, {'candidate_id': 'P2', 'point': [10.001, 10]}]
        old = deepcopy(b);self.assertEqual([p['candidate_id'] for p in geometry_bijection(a, b)], ['P2', 'P1']);self.assertEqual(b, old)

    def test_ambiguous_missing_or_large_move_cannot_pass(self):
        a = [{'candidate_id': 'P1', 'point': [10, 10]}, {'candidate_id': 'P2', 'point': [20, 20]}]
        for b in [[a[0]], [a[0], {'candidate_id': 'P2', 'point': [20.1, 20]}], [a[0], {'candidate_id': 'P2', 'point': [10.01, 10]}]]:
            with self.assertRaises(ValueError):geometry_bijection(a, b)


if __name__ == '__main__':unittest.main()
