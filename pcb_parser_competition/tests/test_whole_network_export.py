from copy import deepcopy
import json
from pathlib import Path
import unittest

from pcb_parser.whole_network_export import recover_interface_titles, net_export_guard, apply_net_export_guard
from pcb_parser.nearby_tip_fields import reconcile_nearby_tip_fields
from pcb_parser.pin_output_rules import sync_fields
from test_nearby_tip_fields import fixture as nearby_fixture


def config():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/whole_network_export_v1.json').read_text(encoding='utf-8'))


def word(text, box, views=3):
    return {'text': text, 'bbox': box, 'confidence': .999, 'view': 'global',
            'observations': [{'text': text, 'bbox': box, 'confidence': .999, 'view': 'view'+str(i)} for i in range(views-1)]}


def interface_fixture():
    record = {'detection_index': 0, 'component_id': 'UNRESOLVED_1', 'designator': None, 'Name': None,
              'type': 'ic', 'bbox_top_left': [40, 20, 80, 80], 'fields': {}}
    ps = [{'candidate_id': 'P'+str(i), 'point': [40, 30+i*5], 'owner': 'UNRESOLVED_1', 'owner_status': 'assigned',
           'owner_detection_index': 0, 'number': str(i), 'pinname': 'NAME'+str(i), 'number_status': 'observed_OCR_text'} for i in range(1, 7)]
    pins = {'size': [120, 120], 'components': [deepcopy(record)], 'pin_candidates': ps}
    graph = {'pin_attachments': [{k: p[k] for k in ['candidate_id', 'point', 'owner', 'owner_status', 'number', 'pinname']} for p in ps],
             'networks': [{'network_id': 'N1', 'pin_candidate_ids': ['P1', 'P2'], 'official_pin_references': [], 'edge_ids': []}], 'edges': []}
    texts = [word('JTAG', [42, 5, 64, 18]), word('JTAG-20PIN', [42, 81, 77, 90], 2)]
    return pins, graph, [record], texts, config()


class InterfaceTitleTests(unittest.TestCase):
    def test_observed_title_updates_identity_and_all_refs_without_guessing_pins(self):
        f = interface_fixture();old = deepcopy(f)
        p, g, r, audit = recover_interface_titles(*f)
        self.assertEqual(audit['new_identities'], 1);self.assertEqual(r[0]['component_id'], 'JTAG')
        self.assertEqual(r[0]['Name'], 'JTAG-20PIN');self.assertEqual(g['networks'][0]['official_pin_references'], ['JTAG.1', 'JTAG.2'])
        self.assertEqual(f, old)
        for a, b in zip(p['pin_candidates'], old[0]['pin_candidates']):
            self.assertEqual((a['point'], a['number'], a['pinname']), (b['point'], b['number'], b['pinname']))
        self.assertEqual(g['networks'][0]['pin_candidate_ids'], ['P1', 'P2'])

    def test_title_alone_does_not_become_identity(self):
        f = interface_fixture();f[3].pop()
        self.assertEqual(recover_interface_titles(*f)[3]['new_identities'], 0)

    def test_single_view_header_rejected(self):
        f = interface_fixture();f[3][0] = word('JTAG', [42, 5, 64, 18], 1)
        self.assertEqual(recover_interface_titles(*f)[3]['new_identities'], 0)

    def test_wrong_footer_prefix_rejected(self):
        f = interface_fixture();f[3][1] = word('OTHER-20PIN', [42, 81, 77, 90], 2)
        self.assertEqual(recover_interface_titles(*f)[3]['new_identities'], 0)

    def test_number_exceeds_observed_pin_count_rejected(self):
        f = interface_fixture();f[0]['pin_candidates'][-1]['number'] = '25';f[1]['pin_attachments'][-1]['number'] = '25'
        _, _, _, audit = recover_interface_titles(*f)
        self.assertEqual(audit['new_identities'], 0);self.assertIn('observed_numbers_conflict_with_footer', audit['withheld'][0]['withheld_reasons'])

    def test_missing_number_evidence_blocks_interface_rule(self):
        f = interface_fixture();f[0]['pin_candidates'][0]['number_status'] = 'inferred'
        self.assertEqual(recover_interface_titles(*f)[3]['new_identities'], 0)

    def test_repeated_title_at_other_location_rejects_no_suffix_created(self):
        f = interface_fixture();f[3].append(word('JTAG', [85, 5, 110, 18]))
        p, _, r, audit = recover_interface_titles(*f)
        self.assertIsNone(r[0]['designator']);self.assertEqual(audit['new_identities'], 0)
        self.assertIn('repeated_title_glyph_source_order_unknown', audit['withheld'][0]['withheld_reasons'])

    def test_existing_model_name_preserved(self):
        f = interface_fixture();f[2][0]['Name'] = 'EXISTING'
        self.assertEqual(recover_interface_titles(*f)[2][0]['Name'], 'EXISTING')

    def test_footer_inside_pin_rows_not_accepted(self):
        f = interface_fixture();f[3][1] = word('JTAG-20PIN', [42, 40, 77, 48], 2)
        self.assertEqual(recover_interface_titles(*f)[3]['new_identities'], 0)

    def test_reapplication_preserves_observed_identity(self):
        p, g, r, audit = recover_interface_titles(*interface_fixture())
        p2, g2, r2, a2 = recover_interface_titles(p, g, r, interface_fixture()[3], config())
        self.assertEqual((p, g, r), (p2, g2, r2));self.assertEqual(a2['new_identities'], 0)


class WholeNetTests(unittest.TestCase):
    def aliased(self):
        p, g, r, c = nearby_fixture();p, g, _ = reconcile_nearby_tip_fields(p, g, r, c)
        g['edges'][0].update(source='A', target='B')
        return p, g, r, config(), c

    def test_two_tips_one_contact_is_withheld(self):
        f = self.aliased();guard = net_export_guard(*f)
        self.assertEqual(guard['summary']['verified_duplicate_tip_pairs'], 1)
        self.assertIn('single_distinct_contact_fragment', guard['blocks'][0]['reasons'])
        self.assertEqual(guard['networks'][0]['original_member_ids'], ['P1', 'P2'])

    def test_same_name_without_revalidated_alias_proof_keeps_both_contacts(self):
        f = self.aliased();f[0]['pin_candidates'][0]['text_evidence']['pinname']['source'] = 'not_a_verified_alias'
        guard = net_export_guard(*f)
        self.assertEqual(guard['summary']['verified_duplicate_tip_pairs'], 0)
        self.assertEqual(guard['networks'][0]['canonical_contact_ids'], ['P1', 'P2'])

    def test_missing_component_contact_at_wire_endpoint_is_reported(self):
        f = self.aliased();f[2].append({'detection_index': 1, 'component_id': 'U2', 'type': 'ic', 'bbox_top_left': [7, 45, 12, 55], 'detection_confidence': .99})
        f[0]['components'].append(deepcopy(f[2][-1]))
        guard = net_export_guard(*f)
        self.assertIn('wire_endpoint_near_component_without_member', guard['blocks'][0]['reasons'])
        self.assertEqual(guard['networks'][0]['unrepresented_component_endpoints'][0]['component_id'], 'U2')

    def test_alias_cannot_make_single_contact_self_loop_export(self):
        f = self.aliased();guard = net_export_guard(*f)
        payload = {'components': {}, 'pins': {'U1': {'pin_8': {'pinname': 'INT'}}}, 'nets': {'L1': {'hyperGraph': '(U1.8)', 'edges': {}}}}
        cov = {'withheld_networks': []}
        out, cov = apply_net_export_guard(payload, cov, guard, f[0], f[1])
        self.assertEqual(out['nets'], {});self.assertEqual(cov['exported_nets'], 0)

    def connected_fixture(self):
        p, g, r, cfg, nc = self.aliased()
        record = {'detection_index': 1, 'component_id': 'U2', 'designator': 'U2', 'type': 'ic', 'bbox_top_left': [0, 30, 10, 70], 'detection_confidence': .99}
        r.append(record);p['components'].append(deepcopy(record))
        other = {'candidate_id': 'P3', 'point': [10, 50], 'owner': 'U2', 'owner_status': 'assigned', 'owner_detection_index': 1,
                 'number': '1', 'pinname': 'SIGNAL', 'confidence': .99, 'text_side': 'right'}
        p['pin_candidates'].append(other)
        a = {k: other[k] for k in ['candidate_id', 'point', 'owner', 'owner_status', 'number', 'pinname']}
        a.update(attachment_status='attached', physical_network_id='N1', network_id='L1', wire_projection=[10, 50], wire_gap_px=0, alternatives=[])
        g['pin_attachments'].append(a)
        for net in g['networks']+g['physical_networks']:net['pin_candidate_ids'].append('P3')
        payload = {'components': {}, 'pins': {'U1': {'pin_8': {'pinname': 'INT'}}, 'U2': {'pin_1': {'pinname': 'SIGNAL'}}}, 'nets': {}}
        cov = {'withheld_networks': [{'network_id': 'L1', 'reasons': ['contains_pin_with_unresolved_identity_number_or_name']}]}
        return p, g, r, cfg, nc, payload, cov

    def test_verified_alias_preserves_all_members_in_complete_two_contact_net(self):
        p, g, r, cfg, nc, payload, cov = self.connected_fixture();old = deepcopy((p, g, payload, cov))
        guard = net_export_guard(p, g, r, cfg, nc);out, cov = apply_net_export_guard(payload, cov, guard, p, g)
        self.assertEqual(out['nets']['L1']['hyperGraph'], '(U1.8,U2.1)')
        proof = cov['atomic_complete_network_proofs'][0]
        self.assertEqual(proof['all_original_member_ids'], ['P1', 'P2', 'P3']);self.assertEqual(proof['member_to_contact']['P2'], 'P1')
        self.assertEqual(proof['distinct_contact_count'], 2);self.assertEqual((p, g, payload), old[:3])

    def test_unproven_unresolved_member_prevents_complete_export(self):
        p, g, r, cfg, nc, payload, cov = self.connected_fixture();p['pin_candidates'][0]['text_evidence']['pinname']['source'] = 'unverified'
        guard = net_export_guard(p, g, r, cfg, nc);out, cov = apply_net_export_guard(payload, cov, guard, p, g)
        self.assertEqual(out['nets'], {})

    def test_alias_does_not_override_junction_gate(self):
        p, g, r, cfg, nc, payload, cov = self.connected_fixture()
        cov['withheld_networks'][0]['reasons'].append('junction_semantics_unresolved')
        guard = net_export_guard(p, g, r, cfg, nc);out, cov = apply_net_export_guard(payload, cov, guard, p, g)
        self.assertEqual(out['nets'], {})

    def test_exported_net_cannot_omit_raw_member_without_alias(self):
        p, g, r, cfg, nc, payload, cov = self.connected_fixture();p['pin_candidates'][0]['text_evidence']['pinname']['source'] = 'unverified'
        payload['nets']['L1'] = {'hyperGraph': '(U1.8,U2.1)', 'edges': {}}
        guard = net_export_guard(p, g, r, cfg, nc)
        with self.assertRaisesRegex(ValueError, 'omitted an unresolved member'):
            apply_net_export_guard(payload, cov, guard, p, g)

    def test_guard_is_idempotent(self):
        p, g, r, cfg, nc, payload, cov = self.connected_fixture()
        guard = net_export_guard(p, g, r, cfg, nc);a, ac = apply_net_export_guard(payload, cov, guard, p, g)
        b, bc = apply_net_export_guard(a, ac, guard, p, g)
        self.assertEqual((a, ac), (b, bc))

    def test_endpoint_coordinates_follow_nodes_when_edge_order_is_reversed(self):
        f = self.aliased();edge = f[1]['edges'][0]
        f[1]['nodes'] = [{'node_id': 'A', 'point': edge['polyline'][-1]}, {'node_id': 'B', 'point': edge['polyline'][0]}]
        f[2].append({'detection_index': 1, 'component_id': 'U2', 'type': 'ic',
            'bbox_top_left': [7, 45, 12, 55], 'detection_confidence': .99})
        f[0]['components'].append(deepcopy(f[2][-1]))
        guard = net_export_guard(*f)
        missing = [m for m in guard['networks'][0]['unrepresented_component_endpoints'] if m['component_id'] == 'U2']
        self.assertEqual(missing[0]['node_id'], 'B');self.assertEqual(missing[0]['point'], edge['polyline'][0])


if __name__ == '__main__':unittest.main()
