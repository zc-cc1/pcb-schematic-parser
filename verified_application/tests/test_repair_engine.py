"""Tests of failure modes that matter when adding new structural rules."""
import copy
import unittest

from pcb_parser.repair_engine import Edit, Proposal, RepairContext, RepairPipeline, RuleSpec


def baseline():
    return {'components': {o: {'type': 'box', 'bbox': [1, 1, 10, 10]} for o in ('A', 'B', 'C')},
            'pins': {o: {'pin_1': {'pinname': 'OUT', 'point': {'x': 5., 'y': 5.}}} for o in ('A', 'B', 'C')},
            'nets': {'N': {'hyperGraph': '(A.1)', 'edges': {'e': [{'x': 2., 'y': 5.}, {'x': 8., 'y': 5.}]}},
                     'M': {'hyperGraph': '(C.1)', 'edges': {'e': [{'x': 2., 'y': 8.}, {'x': 8., 'y': 8.}]}}}}


class Rule:
    def __init__(self, rule_id, build, capabilities=('add_memberships',), enabled=True):
        self.spec = RuleSpec(rule_id, 'test', 'test rule', capabilities, enabled)
        self.build = build

    def propose(self, context):
        return self.build(context, self.spec.rule_id)


def attach(context, rule_id, network='N'):
    old = context.target['nets'][network]['hyperGraph']
    after = old[:-1] + ',B.1)'
    return [Proposal(rule_id, (Edit.set(context.target, ('nets', network, 'hyperGraph'), after),),
                     {'observation': 'unique wire'})]


class RepairEngineTests(unittest.TestCase):
    def run_rules(self, rules, target=None, **kwargs):
        target = baseline() if target is None else target
        return RepairPipeline(rules).run(RepairContext(target, {}, 100, 100), **kwargs)

    def test_conflicting_network_writes_withhold_both(self):
        rules = [Rule('first', attach), Rule('second', attach)]
        out, audit = self.run_rules(rules)
        self.assertEqual(out, baseline())
        self.assertEqual(len(audit['withheld']), 2)
        self.assertTrue(all(p['rejection'] == 'conflicting_proposals' for p in audit['proposals']))

    def test_same_pin_claimed_by_two_networks_withholds_both(self):
        out, audit = self.run_rules([Rule('first', attach), Rule('second', lambda c, r: attach(c, r, 'M'))])
        self.assertEqual(out, baseline())
        self.assertEqual(len(audit['withheld']), 2)

    def test_conflicts_independent_of_rule_registration_order(self):
        rules = [Rule('first', attach), Rule('second', lambda c, r: attach(c, r, 'M'))]
        self.assertEqual(self.run_rules(rules), self.run_rules(list(reversed(rules))))

    def test_duplicate_identical_proposal_commits_once(self):
        out, audit = self.run_rules([Rule('same', lambda c, r: attach(c, r) * 2)])
        self.assertEqual(len(audit['accepted']), 1)
        self.assertEqual(audit['contact_count'], 1)

    def test_global_budget_and_atomic_branch_pin(self):
        def build(context, rule_id):
            t = context.target
            pin = {'pinname': '', 'point': {'x': 7., 'y': 7.}}
            branch = {'hyperGraph': '(B.1,B.2)', 'edges': {'e': [{'x': 5., 'y': 5.}, {'x': 7., 'y': 7.}]}}
            return [Proposal(rule_id, (Edit.set(t, ('pins', 'B', 'pin_2'), pin), Edit.set(t, ('nets', 'NEW'), branch)))]
        rule = Rule('branch', build, ('add_networks', 'add_pins'))
        out, audit = self.run_rules([rule], maximum_new_contacts=1)
        self.assertEqual(out, baseline())
        self.assertEqual(audit['proposals'][0]['rejection'], 'contact_budget')
        out, audit = self.run_rules([rule])
        self.assertIn('pin_2', out['pins']['B'])
        self.assertIn('NEW', out['nets'])

    def test_invalid_pin_rolls_back_whole_branch(self):
        def build(context, rule_id):
            t = context.target
            return [Proposal(rule_id, (Edit.set(t, ('pins', 'B', 'pin_2'), {'pinname': '', 'point': {'x': -1., 'y': 5.}}),
                                      Edit.set(t, ('nets', 'N', 'hyperGraph'), '(A.1,B.2)')))]
        out, audit = self.run_rules([Rule('invalid', build, ('add_memberships', 'add_pins'))])
        self.assertEqual(out, baseline())
        self.assertEqual(audit['proposals'][0]['rejection'], 'invalid_new_pin_point')

    def test_invalid_new_wire_rolls_back(self):
        def build(context, rule_id):
            net = {'hyperGraph': '(B.1)', 'edges': {'e': [{'x': 0., 'y': 0.}, {'x': float('inf'), 'y': 0.}]}}
            return [Proposal(rule_id, (Edit.set(context.target, ('nets', 'NEW'), net),))]
        out, audit = self.run_rules([Rule('invalid', build, ('add_networks',))])
        self.assertEqual(out, baseline())
        self.assertTrue(audit['errors'])  # Non-finite values cannot enter audit JSON.

    def test_generator_failure_discards_all_its_proposals(self):
        def broken(context, rule_id):
            yield from attach(context, rule_id)
            raise RuntimeError('failed after yield')
        out, audit = self.run_rules([Rule('broken', broken)])
        self.assertEqual(out, baseline())
        self.assertFalse(audit['proposals'])
        self.assertEqual(audit['errors'][0]['rule_id'], 'broken')

    def test_failed_rule_does_not_block_independent_rule(self):
        def broken(context, rule_id):
            raise RuntimeError('failed')
        out, audit = self.run_rules([Rule('broken', broken), Rule('working', attach)])
        self.assertEqual(audit['contact_count'], 1)
        self.assertEqual(len(audit['errors']), 1)

    def test_rule_cannot_mutate_target(self):
        def bad(context, rule_id):
            context.target['pins'].clear()
            return []
        target = baseline()
        out, audit = self.run_rules([Rule('bad', bad)], target)
        self.assertEqual(out, target)
        self.assertEqual(target, baseline())
        self.assertIn('rule_mutated_target', audit['errors'][0]['reason'])

    def test_evidence_is_isolated_between_rules(self):
        def bad(context, rule_id):
            context.evidence['items'].clear()
            return []
        def observer(context, rule_id):
            self.assertEqual(context.evidence['items'], [1, 2])
            return attach(context, rule_id)
        evidence = {'items': [1, 2]}
        out, audit = RepairPipeline([Rule('a', bad), Rule('b', observer)]).run(RepairContext(baseline(), evidence, 100, 100))
        self.assertEqual(evidence, {'items': [1, 2]})
        self.assertFalse(audit['errors'])
        self.assertEqual(audit['contact_count'], 1)

    def test_stale_preconditions_are_rejected(self):
        def stale(context, rule_id):
            edit = Edit(('nets', 'N', 'hyperGraph'), True, '(WRONG.1)', '(A.1,B.1)')
            return [Proposal(rule_id, (edit,))]
        out, audit = self.run_rules([Rule('stale', stale)])
        self.assertEqual(out, baseline())
        self.assertEqual(audit['proposals'][0]['rejection'], 'stale_precondition')

    def test_dangling_member_is_rejected(self):
        def build(context, rule_id):
            return [Proposal(rule_id, (Edit.set(context.target, ('nets', 'N', 'hyperGraph'), '(A.1,UNKNOWN.7)'),))]
        out, audit = self.run_rules([Rule('invalid', build)])
        self.assertEqual(out, baseline())
        self.assertEqual(audit['proposals'][0]['rejection'], 'dangling_member')

    def test_no_connect_is_rejected_centrally(self):
        target = baseline()
        target['pins']['B']['pin_1']['pinname'] = ' N.C. '
        out, audit = self.run_rules([Rule('nc', attach)], target)
        self.assertEqual(out, target)
        self.assertEqual(audit['proposals'][0]['rejection'], 'no_connect_member')

    def test_already_assigned_pin_cannot_be_added_elsewhere(self):
        target = baseline()
        target['nets']['M']['hyperGraph'] = '(B.1,C.1)'
        out, audit = self.run_rules([Rule('assigned', attach)], target)
        self.assertEqual(out, target)
        self.assertEqual(audit['proposals'][0]['rejection'], 'member_already_on_another_network')

    def test_cannot_rewrite_component_or_existing_pin(self):
        for path, value in [(('components', 'A', 'type'), 'gnd'),
                            (('pins', 'A', 'pin_1', 'point'), {'x': 8., 'y': 8.})]:
            def build(context, rule_id):
                return [Proposal(rule_id, (Edit.set(context.target, path, value),))]
            out, audit = self.run_rules([Rule('rewrite', build)])
            self.assertEqual(out, baseline())
            self.assertEqual(audit['proposals'][0]['rejection'], 'unsupported_edit_requires_new_validator')

    def test_cannot_remove_existing_membership(self):
        def build(context, rule_id):
            return [Proposal(rule_id, (Edit.set(context.target, ('nets', 'N', 'hyperGraph'), '(B.1)'),))]
        out, audit = self.run_rules([Rule('remove', build)])
        self.assertEqual(out, baseline())
        self.assertEqual(audit['proposals'][0]['rejection'], 'unsupported_edit_requires_new_validator')

    def test_rule_capabilities_are_enforced(self):
        out, audit = self.run_rules([Rule('wrong_capability', attach, ('add_pins',))])
        self.assertEqual(out, baseline())
        self.assertEqual(audit['proposals'][0]['rejection'], 'rule_capability_denied')

    def test_duplicate_rule_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate_rule_id'):
            RepairPipeline([Rule('same', attach), Rule('same', attach)])

    def test_rule_identity_cannot_be_spoofed(self):
        out, audit = self.run_rules([Rule('first', lambda c, r: attach(c, 'another'))])
        self.assertEqual(out, baseline())
        self.assertTrue(audit['errors'])

    def test_disabled_rule_never_executes(self):
        def fail(context, rule_id):
            self.fail('Disabled rule was executed')
        out, audit = self.run_rules([Rule('off', fail)], rule_settings={'off': {'enabled': False}})
        self.assertEqual(out, baseline())
        self.assertFalse(audit['errors'])

    def test_unknown_rule_configuration_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unknown_rule_configuration'):
            self.run_rules([Rule('known', attach)], rule_settings={'typo': {'enabled': False}})

    def test_invalid_enabled_configuration_rejected(self):
        with self.assertRaisesRegex(ValueError, 'rule_enabled_must_be_boolean'):
            self.run_rules([Rule('known', attach)], rule_settings={'known': {'enabled': 'false'}})

    def test_negative_fractional_or_boolean_budget_rejected(self):
        for value in (-1, 1.5, True):
            with self.assertRaisesRegex(ValueError, 'budgets_must_be_nonnegative_integers'):
                self.run_rules([Rule('known', attach)], maximum_new_contacts=value)

    def test_delete_empty_net_conflicts_with_descendant_write(self):
        target = baseline()
        target['nets']['N']['edges'] = {}
        def remove(context, rule_id):
            return [Proposal(rule_id, (Edit.remove(context.target, ('nets', 'N')),))]
        out, audit = self.run_rules([Rule('remove', remove, ('remove_empty_networks',)), Rule('attach', attach)], target)
        self.assertEqual(out, target)
        self.assertEqual(len(audit['withheld']), 2)

    def test_delete_network_with_wires_is_rejected(self):
        def remove(context, rule_id):
            return [Proposal(rule_id, (Edit.remove(context.target, ('nets', 'N')),))]
        out, audit = self.run_rules([Rule('remove', remove, ('remove_empty_networks',))])
        self.assertEqual(out, baseline())
        self.assertEqual(audit['proposals'][0]['rejection'], 'unsupported_edit_requires_new_validator')

    def test_overlapping_edits_inside_proposal_are_rejected(self):
        def build(context, rule_id):
            return [Proposal(rule_id, (Edit.set(context.target, ('nets', 'N', 'hyperGraph'), '(A.1,B.1)'),
                                      Edit.set(context.target, ('nets', 'N', 'hyperGraph'), '(A.1,C.1)')))]
        out, audit = self.run_rules([Rule('overlap', build)])
        self.assertEqual(out, baseline())
        self.assertEqual(audit['proposals'][0]['rejection'], 'overlapping_edits_in_proposal')


if __name__ == '__main__':
    unittest.main()
