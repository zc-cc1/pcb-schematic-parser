from copy import deepcopy
import json
from pathlib import Path
import unittest

from pcb_parser.engineering_network_export import complete_engineering_fields
from pcb_parser.pin_output_rules import sync_fields
from pcb_parser.whole_network_export import net_export_guard, apply_net_export_guard
from run_network_labels import logical_partial_export


ROOT = Path(__file__).resolve().parents[1]
def config(name='engineering_network_export_v1.json'):
    return json.loads((ROOT/'configs'/name).read_text(encoding='utf-8'))


def fixture(kind='resistor', box=None):
    r = {'component_id': 'R1', 'detection_index': 0, 'designator': 'R1', 'type': kind,
         'Name': None, 'value': None, 'bbox_top_left': box or [40, 45, 60, 55], 'detection_confidence': .99}
    c = {**r, 'confidence': .99}
    p = {'size': [100, 100], 'components': [c], 'pin_candidates': [
        {'candidate_id': 'P'+str(i), 'point': point, 'confidence': .99, 'owner': 'R1',
         'owner_status': 'assigned', 'owner_detection_index': 0, 'number': None, 'pinname': None,
         'text_evidence': {}, 'text_candidates': {'number': [], 'pinname': []}}
        for i, point in [(1, [40, 50]), (2, [60, 50])]]}
    g = {'size': [100, 100], 'edges': [
        {'edge_id': 'E1', 'source': 'A', 'target': 'B', 'polyline': [[10, 50], [40, 50]], 'mean_probability': .99},
        {'edge_id': 'E2', 'source': 'C', 'target': 'D', 'polyline': [[60, 50], [90, 50]], 'mean_probability': .99}],
        'pin_attachments': [{'candidate_id': 'P'+str(i), 'point': point, 'attachment_status': 'attached',
            'physical_network_id': 'N'+str(i), 'network_id': 'L'+str(i), 'wire_projection': point,
            'wire_gap_px': 0, 'alternatives': []} for i, point in [(1, [40, 50]), (2, [60, 50])]],
        'physical_networks': [{'network_id': 'N'+str(i), 'edge_ids': ['E'+str(i)], 'pin_candidate_ids': ['P'+str(i)]} for i in [1, 2]],
        'networks': [{'network_id': 'L'+str(i), 'edge_ids': ['E'+str(i)], 'pin_candidate_ids': ['P'+str(i)],
            'unresolved_junction_ids': [], 'flywire': False, 'official_pin_references': []} for i in [1, 2]],
        'gap_candidates': [], 'summary': {'logical_label_merges': 0}}
    return p, sync_fields(p, g), [r], config()


class EngineeringFieldTests(unittest.TestCase):
    def test_two_opposite_contacts_generate_only_tagged_fields(self):
        f = fixture();old = deepcopy(f);p, g, a = complete_engineering_fields(*f)
        self.assertEqual([(x['number'], x['pinname']) for x in p['pin_candidates']], [('1', '1'), ('2', '2')])
        self.assertEqual(a['summary']['generated_pin_numbers'], 2);self.assertEqual(a['summary']['generated_pin_names'], 2)
        self.assertFalse(p['pin_candidates'][0]['text_evidence']['number']['observed_OCR_glyph'])
        self.assertEqual(f, old);self.assertEqual(g['edges'], old[1]['edges']);self.assertEqual(g['physical_networks'], old[1]['physical_networks'])

    def test_capacitor_axis_comes_from_leads_even_when_body_is_taller(self):
        f = fixture('capacitor', [40, 30, 60, 70]);p, _, a = complete_engineering_fields(*f)
        self.assertEqual(a['summary']['generated_pin_numbers'], 2);self.assertEqual(p['pin_candidates'][0]['number'], '1')

    def test_top_bottom_convention_uses_image_axis(self):
        f = fixture();f[2][0]['bbox_top_left'] = [45, 40, 55, 60];f[0]['components'][0]['bbox_top_left'] = [45, 40, 55, 60]
        for i, pin in enumerate(f[0]['pin_candidates']):
            pin['point'] = [50, 40+20*i];f[1]['pin_attachments'][i].update(point=pin['point'], wire_projection=pin['point'])
        f[1]['edges'][0]['polyline'] = [[50, 10], [50, 40]];f[1]['edges'][1]['polyline'] = [[50, 60], [50, 90]]
        p, _, _ = complete_engineering_fields(*f);self.assertEqual([x['number'] for x in p['pin_candidates']], ['1', '2'])

    def test_adjacent_open_wire_is_withdrawn_without_dropping_member(self):
        f = fixture();f[0]['pin_candidates'][1]['point'] = [40, 59]
        f[1]['pin_attachments'][1].update(point=[40, 59], wire_projection=[40, 59]);f[1]['edges'][1]['polyline'] = [[10, 59], [40, 59]]
        p, g, a = complete_engineering_fields(*f)
        self.assertEqual(a['summary']['blank_owner_assignments_withdrawn'], 1);self.assertIsNone(p['pin_candidates'][1]['owner'])
        self.assertEqual(a['summary']['generated_pin_numbers'], 0);self.assertEqual(g['networks'][1]['pin_candidate_ids'], ['P2'])
        self.assertEqual(g['pin_attachments'][1]['owner_status'], 'engineering_geometry_withheld')

    def test_existing_field_on_misaligned_tip_is_protected(self):
        f = fixture();f[0]['pin_candidates'][1].update(point=[40, 59], number='2')
        f[1]['pin_attachments'][1].update(point=[40, 59], wire_projection=[40, 59]);f[1]['edges'][1]['polyline'] = [[10, 59], [40, 59]]
        p, _, a = complete_engineering_fields(*f)
        self.assertEqual(p['pin_candidates'][1]['owner'], 'R1');self.assertEqual(p['pin_candidates'][1]['number'], '2')
        self.assertEqual(len(a['existing_fields_protected']), 1);self.assertEqual(a['summary']['generated_pin_numbers'], 0)

    def test_observed_reversed_number_is_not_overwritten_or_reinterpreted(self):
        f = fixture();f[0]['pin_candidates'][0]['number'] = '2';p, _, a = complete_engineering_fields(*f)
        self.assertEqual(p['pin_candidates'][0]['number'], '2');self.assertEqual(a['summary']['generated_pin_numbers'], 0)

    def test_existing_matching_number_can_fill_name_but_retains_source(self):
        f = fixture();f[0]['pin_candidates'][0].update(number='1', number_status='observed_OCR_text')
        f[0]['pin_candidates'][0]['text_evidence']['number'] = {'source': 'old_observation'}
        p, _, a = complete_engineering_fields(*f)
        self.assertEqual(a['summary']['generated_pin_numbers'], 1);self.assertEqual(p['pin_candidates'][0]['text_evidence']['number']['source'], 'old_observation')

    def test_two_tips_same_wire_cannot_be_two_terminals(self):
        f = fixture();f[1]['pin_attachments'][1]['physical_network_id'] = 'N1';f[1]['physical_networks'][0]['edge_ids'].append('E2')
        self.assertEqual(complete_engineering_fields(*f)[2]['summary']['generated_pin_numbers'], 0)

    def test_missing_second_tip_is_never_fabricated(self):
        f = fixture();f[0]['pin_candidates'].pop();f[1]['pin_attachments'].pop();f[1]['networks'][1]['pin_candidate_ids'] = []
        p, _, a = complete_engineering_fields(*f);self.assertEqual(len(p['pin_candidates']), 1);self.assertEqual(a['summary']['generated_pin_numbers'], 0)

    def test_extra_tip_on_passive_requires_review(self):
        f = fixture();third = deepcopy(f[0]['pin_candidates'][0]);third['candidate_id'] = 'P3';f[0]['pin_candidates'].append(third)
        att = deepcopy(f[1]['pin_attachments'][0]);att['candidate_id'] = 'P3';f[1]['pin_attachments'].append(att);f[1]['networks'][0]['pin_candidate_ids'].append('P3')
        self.assertEqual(complete_engineering_fields(*f)[2]['summary']['generated_pin_numbers'], 0)

    def test_local_wire_branch_does_not_force_axis(self):
        f = fixture();f[1]['edges'].append({'edge_id': 'E3', 'polyline': [[40, 30], [40, 50]], 'mean_probability': .99})
        f[1]['physical_networks'][0]['edge_ids'].append('E3')
        self.assertEqual(complete_engineering_fields(*f)[2]['summary']['generated_pin_numbers'], 0)

    def test_close_number_candidates_block_convention(self):
        f = fixture();f[0]['pin_candidates'][0]['text_candidates']['number'] = [{'text': '3', 'cost': 8}]
        self.assertEqual(complete_engineering_fields(*f)[2]['summary']['generated_pin_numbers'], 0)

    def test_package_code_candidates_do_not_become_pin_numbers(self):
        f = fixture();f[0]['pin_candidates'][0]['text_candidates']['number'] = [{'text': '0402', 'cost': 8}]
        self.assertEqual(complete_engineering_fields(*f)[2]['summary']['generated_pin_numbers'], 2)

    def test_no_generated_identity_for_unknown_passive(self):
        f = fixture();f[2][0]['designator'] = None
        self.assertEqual(complete_engineering_fields(*f)[2]['summary']['generated_pin_numbers'], 0)

    def test_ic_and_four_terminal_crystal_excluded(self):
        for kind in ['ic', 'crystal']:
            self.assertEqual(complete_engineering_fields(*fixture(kind))[2]['summary']['generated_pin_numbers'], 0)

    def test_low_confidence_or_missing_attachment_blocks_pair(self):
        for which in ['tip', 'wire', 'detection', 'gap']:
            f = fixture()
            if which == 'tip':f[0]['pin_candidates'][0]['confidence'] = .5
            if which == 'wire':f[1]['edges'][0]['mean_probability'] = .5
            if which == 'detection':f[2][0]['detection_confidence'] = .5
            if which == 'gap':f[1]['pin_attachments'][0]['wire_gap_px'] = None
            self.assertEqual(complete_engineering_fields(*f)[2]['summary']['generated_pin_numbers'], 0)

    def test_reapplication_is_stable(self):
        p, g, a = complete_engineering_fields(*fixture());p2, g2, a2 = complete_engineering_fields(p, g, fixture()[2], config())
        self.assertEqual((p, g), (p2, g2));self.assertEqual(a2['changes'], [])

    def test_atomic_export_keeps_unresolved_real_members(self):
        f = fixture();other = {'candidate_id': 'P3', 'point': [10, 50], 'confidence': .99, 'owner': None,
            'owner_status': 'unresolved', 'owner_detection_index': None, 'number': None, 'pinname': None}
        f[0]['pin_candidates'].append(other);att = deepcopy(f[1]['pin_attachments'][0]);att.update(candidate_id='P3', point=[10, 50], wire_projection=[10, 50])
        f[1]['pin_attachments'].append(att);f[1]['networks'][0]['pin_candidate_ids'].append('P3')
        p, g, _ = complete_engineering_fields(*f);payload, cov = logical_partial_export(p, g, f[2])
        guard = net_export_guard(p, g, f[2], config('whole_network_export_v1.json'), config('nearby_tip_fields_v1.json'))
        out, _ = apply_net_export_guard(payload, cov, guard, p, g)
        self.assertEqual(out['nets'], {});self.assertEqual(g['networks'][0]['pin_candidate_ids'], ['P1', 'P3'])

    def connected(self):
        f = fixture();record = {'component_id': 'U1', 'detection_index': 1, 'designator': 'U1', 'type': 'ic',
            'Name': None, 'value': None, 'bbox_top_left': [0, 40, 10, 60], 'detection_confidence': .99}
        f[2].append(record);f[0]['components'].append(deepcopy(record))
        other = {'candidate_id': 'P3', 'point': [10, 50], 'confidence': .99, 'owner': 'U1', 'owner_status': 'assigned',
            'owner_detection_index': 1, 'number': '1', 'pinname': 'SIGNAL'}
        f[0]['pin_candidates'].append(other);att = deepcopy(f[1]['pin_attachments'][0])
        att.update(candidate_id='P3', point=[10, 50], wire_projection=[10, 50]);f[1]['pin_attachments'].append(att)
        f[1]['networks'][0]['pin_candidate_ids'].append('P3');f[1]['physical_networks'][0]['pin_candidate_ids'].append('P3')
        return f

    def test_whole_net_resolves_generated_and_observed_contacts(self):
        f = self.connected();p, g, _ = complete_engineering_fields(*f);out, cov = logical_partial_export(p, g, f[2])
        guard = net_export_guard(p, g, f[2], config('whole_network_export_v1.json'), config('nearby_tip_fields_v1.json'))
        out, cov = apply_net_export_guard(out, cov, guard, p, g)
        self.assertEqual(out['nets']['L1']['hyperGraph'], '(R1.1,U1.1)')
        self.assertEqual(cov['atomic_complete_network_proofs'][0]['all_original_member_ids'], ['P1', 'P3'])
        self.assertEqual(out['pins']['U1']['pin_1']['pinname'], 'SIGNAL')

    def test_convention_does_not_bypass_junction_or_gap(self):
        for blocker in ['junction', 'gap']:
            f = self.connected()
            if blocker == 'junction':f[1]['networks'][0]['unresolved_junction_ids'] = ['J1']
            else:f[1]['gap_candidates'] = [{'network_ids': ['L1']}]
            p, g, _ = complete_engineering_fields(*f);out, cov = logical_partial_export(p, g, f[2])
            guard = net_export_guard(p, g, f[2], config('whole_network_export_v1.json'), config('nearby_tip_fields_v1.json'))
            out, _ = apply_net_export_guard(out, cov, guard, p, g);self.assertEqual(out['nets'], {})

    def test_unqualified_second_attachment_never_synthesizes_wire(self):
        f = fixture();f[1]['pin_attachments'][1]['alternatives'] = [{'physical_network_id': 'OTHER'}]
        p, g, a = complete_engineering_fields(*f)
        self.assertEqual(a['summary']['generated_pin_numbers'], 0);self.assertEqual(g['edges'], f[1]['edges'])


if __name__ == '__main__':unittest.main()
