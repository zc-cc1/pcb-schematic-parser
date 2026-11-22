import copy
import unittest

from pcb_parser.connectivity_recovery import recover_connectivity
from test_connectivity_recovery import fixture


class RuleIntegrationTests(unittest.TestCase):
    def test_known_rule_can_be_disabled_independently(self):
        target, graph, pins, rgb = fixture()
        out, audit = recover_connectivity(target, graph, pins, rgb,
            cfg={'rules': {'known_contact_membership': {'enabled': False}}})
        self.assertEqual(out, target)
        self.assertFalse(audit['changes'])

    def test_strings_cannot_silently_enable_destructive_options(self):
        target, graph, pins, rgb = fixture()
        for key in ('prune_zero_wire_fragments', 'restore_clear_branches'):
            with self.assertRaises(ValueError):
                recover_connectivity(target, graph, pins, rgb, cfg={key: 'false'})

    def test_invalid_confidence_and_pixel_limits_rejected(self):
        target, graph, pins, rgb = fixture()
        for config in ({'minimum_power_confidence': float('nan')},
                       {'minimum_power_confidence': 1.1},
                       {'minimum_red_pixels': -1}, {'maximum_new_contacts': 1.5},
                       {'rules': []}):
            with self.assertRaises(ValueError):
                recover_connectivity(target, graph, pins, rgb, cfg=config)

    def test_branch_rule_can_be_disabled_independently(self):
        target, graph, pins, rgb = fixture(False)
        out, audit = recover_connectivity(target, graph, pins, rgb,
            cfg={'rules': {'confirmed_physical_branch': {'enabled': False}}})
        self.assertEqual(out, target)

    def test_legacy_branch_switch_still_disables_rule(self):
        target, graph, pins, rgb = fixture(False)
        out, audit = recover_connectivity(target, graph, pins, rgb,
            cfg={'restore_clear_branches': False, 'rules': {'confirmed_physical_branch': {'enabled': True}}})
        self.assertEqual(out, target)

    def test_disabled_configuration_does_not_leak_to_next_image(self):
        target, graph, pins, rgb = fixture()
        config = {'rules': {'known_contact_membership': {'enabled': False}}}
        old_config = copy.deepcopy(config)
        recover_connectivity(target, graph, pins, rgb, cfg=config)
        out, audit = recover_connectivity(target, graph, pins, rgb)
        self.assertEqual(config, old_config)
        self.assertEqual(audit['new_contact_count'], 1)

    def test_each_accepted_edit_has_evidence_and_version(self):
        target, graph, pins, rgb = fixture()
        out, audit = recover_connectivity(target, graph, pins, rgb)
        execution = audit['rule_execution']
        row = next(p for p in execution['proposals'] if p['status'] == 'accepted')
        self.assertEqual(row['rule_id'], 'known_contact_membership')
        self.assertEqual(row['version'], '1.0')
        self.assertEqual(row['evidence']['contacts'][0]['alternatives'][0]['distance_px'], 0)
        self.assertEqual(len(execution['accepted']), 1)

    def test_rule_rejection_reason_and_no_partial_pin_commit(self):
        target, graph, pins, rgb = fixture(False)
        graph['physical_networks'][0]['unresolved_junction_ids'] = ['uncertain']
        out, audit = recover_connectivity(target, graph, pins, rgb)
        self.assertEqual(out, target)
        row = audit['rule_execution']['proposals'][0]
        self.assertEqual(row['rejection'], 'unresolved_crossings')

    def test_cleanup_is_visible_as_withheld_proposal(self):
        target, graph, pins, rgb = fixture()
        target['components']['PORT']['type'] = 'box'
        target['pins']['PORT']['pin_1']['point']['y'] = 20
        target['nets']['F'] = {'hyperGraph': '(PORT.1)', 'edges': {}}
        out, audit = recover_connectivity(target, graph, pins, rgb)
        row = next(p for p in audit['rule_execution']['proposals'] if p['rule_id'] == 'zero_wire_fragment')
        self.assertEqual(row['rejection'], 'cleanup_not_opted_in')
        self.assertIn('F', out['nets'])


if __name__ == '__main__':
    unittest.main()
