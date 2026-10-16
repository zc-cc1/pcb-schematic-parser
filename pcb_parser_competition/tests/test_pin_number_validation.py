from copy import deepcopy
from pathlib import Path
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'tests'))
from test_pin_output_rules import example
from pcb_parser.pin_number_validation import withhold_package_numbers

CFG = json.loads((ROOT/'configs/pin_number_validation_v1.json').read_text(encoding='utf-8'))


class PinNumberRoleTests(unittest.TestCase):
    def test_package_like_number_withheld_with_original_evidence_retained(self):
        p, g, r = example(number='0402')
        original = deepcopy((p, g, r))
        updated, graph, audit = withhold_package_numbers(p, g, r, CFG)
        pin = updated['pin_candidates'][0]
        self.assertIsNone(pin['number'])
        self.assertEqual(pin['withheld_number_observations'][0]['original_number'], '0402')
        self.assertEqual(audit['changes'][0]['original_evidence'], p['pin_candidates'][0]['text_evidence']['number'])
        self.assertIsNone(graph['pin_attachments'][0]['number'])
        self.assertEqual(graph['edges'], g['edges'])
        self.assertEqual(graph['networks'], g['networks'])
        self.assertEqual((p, g, r), original)

    def test_valid_small_numbers_and_other_literals_not_silently_replaced(self):
        for value in ['1', '2', '10', '01', 'A5', None]:
            p, g, r = example(number=value)
            updated, _, audit = withhold_package_numbers(p, g, r, CFG)
            self.assertEqual(updated, p)
            self.assertEqual(audit['changes'], [])

    def test_IC_and_other_classes_are_not_filtered_by_passive_package_rule(self):
        for kind in ['ic', 'connector', 'crystal', 'diode', 'gnd']:
            p, g, r = example(kind=kind, number='0402')
            updated, _, audit = withhold_package_numbers(p, g, r, CFG)
            self.assertEqual(updated, p)

    def test_library_or_format_number_is_not_treated_as_OCR(self):
        for status in ['public_library_from_observed_pinname', 'PDF_single_tip_label_format_auto_number']:
            p, g, r = example(number='0402')
            p['pin_candidates'][0]['number_status'] = status
            updated, _, audit = withhold_package_numbers(p, g, r, CFG)
            self.assertEqual(updated, p)

    def test_number_without_matching_OCR_evidence_not_filtered(self):
        p, g, r = example(number='0402')
        p['pin_candidates'][0]['text_evidence']['number']['text'] = '1'
        updated, _, audit = withhold_package_numbers(p, g, r, CFG)
        self.assertEqual(updated, p)

    def test_separately_observed_name_is_preserved(self):
        p, g, r = example(number='0402')
        p['pin_candidates'][0].update(pinname='OUT', pinname_status='observed_OCR_text')
        updated, _, audit = withhold_package_numbers(p, g, r, CFG)
        self.assertEqual(updated['pin_candidates'][0]['pinname'], 'OUT')

    def test_dependent_candidate_name_is_withheld_with_package_number(self):
        p, g, r = example(number='0402')
        p['pin_candidates'][0].update(pinname='0402', pinname_status='candidate_PDF_example_number_as_passive_name')
        p['pin_candidates'][0]['text_evidence']['pinname'] = {'provisional': True}
        updated, _, audit = withhold_package_numbers(p, g, r, CFG)
        self.assertIsNone(updated['pin_candidates'][0]['pinname'])
        self.assertEqual(audit['changes'][0]['dependent_candidate_pinname_withheld'], '0402')

    def test_filter_is_idempotent_and_does_not_duplicate_audit_history(self):
        p, g, r = example(number='0402')
        updated, graph, _ = withhold_package_numbers(p, g, r, CFG)
        again, _, audit = withhold_package_numbers(updated, graph, r, CFG)
        self.assertEqual(again, updated)
        self.assertEqual(audit['changes'], [])

    def test_other_pin_geometry_and_owner_are_preserved(self):
        p, g, r = example(number='0603')
        updated, _, audit = withhold_package_numbers(p, g, r, CFG)
        for field in ['candidate_id', 'point', 'owner', 'owner_status', 'owner_detection_index']:
            self.assertEqual(updated['pin_candidates'][0][field], p['pin_candidates'][0][field])

    def test_invalid_policy_is_rejected(self):
        p, g, r = example(number='0402')
        for change in [{'require_observed_OCR_number': False}, {'component_types': ['ic']}, {'package_literals': ['1']}]:
            with self.assertRaises(ValueError):
                withhold_package_numbers(p, g, r, {**CFG, **change})

    def test_cross_owner_contract_is_rejected(self):
        p, g, r = example(number='0402')
        p['pin_candidates'][0]['owner'] = 'WRONG'
        with self.assertRaises(ValueError):
            withhold_package_numbers(p, g, r, CFG)

    def test_withdrawn_number_is_removed_from_derived_reference_strings(self):
        p,g,r=example(number='0402')
        g['networks'][0]['official_pin_references']=['R1.0402','OTHER.1']
        updated,graph,audit=withhold_package_numbers(p,g,r,CFG)
        self.assertEqual(graph['networks'][0]['official_pin_references'],['OTHER.1'])
        self.assertEqual(graph['networks'][0]['pin_candidate_ids'],g['networks'][0]['pin_candidate_ids'])
        self.assertEqual(audit['derived_reference_changes'][0]['before'],['R1.0402','OTHER.1'])
