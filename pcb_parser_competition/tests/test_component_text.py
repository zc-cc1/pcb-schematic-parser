"""Regression cases for ambiguous text, field roles and coordinate transforms."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcb_parser.component_text import bind_components, choose_global, duplicate_detections, merge_texts, restore_polygon, value_kind


def token(text, b, index):
    x1, y1, x2, y2 = b
    return {"text_id": f"text_{index}", "text": text, "confidence": .99, "view": "test",
            "polygon": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]], "bbox": b}


def detection(kind, b):
    return {"class_name": kind, "bbox": b, "confidence": .95}


CFG = {"class_mapping": {"ic": "ic", "capacitor": "capacitor", "resistor": "resistor"}}


class ComponentTextTests(unittest.TestCase):
    def test_rotation_inverse_and_crop_offset(self):
        # Original patch coordinates (12, 20) -> clockwise (79, 12), zoom x2.
        restored = restore_polygon([[158, 24]], [100, 200, 150, 300], 2, 90)
        np.testing.assert_allclose(restored, [[112, 220]])

    def test_same_value_at_two_locations_survives_dedup(self):
        a = token("100n", [5,5,35,15], 1)
        b = token("100n", [105,5,135,15], 2)
        self.assertEqual(len(merge_texts([a, a, b])), 2)

    def test_two_capacitors_keep_separate_repeated_values(self):
        ds = [detection("capacitor", [40,35,50,55]), detection("capacitor", [140,35,150,55])]
        ts = [token("C1", [15,25,35,35], 1), token("100n", [5,38,35,48], 2),
              token("C2", [115,25,135,35], 3), token("100n", [105,38,135,48], 4)]
        output, _, debug = bind_components(ds, ts, 200, 100, CFG)
        self.assertEqual(output["components"]["C1"]["value"], "100n")
        self.assertEqual(output["components"]["C2"]["value"], "100n")
        self.assertEqual(debug["counts"]["resolved_designators"], 2)

    def test_ic_ball_label_does_not_displace_reference(self):
        ds = [detection("ic", [100,40,200,240])]
        ts = [token("U18", [75,170,98,180], 1), token("IC2", [140,245,170,258], 2),
              token("BCM2835", [130,20,180,35], 3)]
        _, records, _ = bind_components(ds, ts, 300, 300, CFG)
        self.assertEqual(records[0]["designator"], "IC2")
        self.assertEqual(records[0]["Name"], "BCM2835")

    def test_ambiguous_ref_is_left_empty(self):
        ds = [detection("resistor", [20,40,30,60]), detection("resistor", [60,40,70,60])]
        ts = [token("R1", [35,40,55,50], 1)]
        _, records, debug = bind_components(ds, ts, 100, 100, CFG)
        self.assertTrue(all(r["designator"] is None for r in records))
        self.assertTrue(debug["rejected_matches"])

    def test_spelling_and_units_are_not_silently_normalized(self):
        ds = [detection("ic", [40,50,100,100])]
        ts = [token("u2", [55,15,75,25], 1), token("ChipA123", [50,30,95,42], 2)]
        output, _, _ = bind_components(ds, ts, 150, 150, CFG)
        self.assertIn("u2", output["components"])
        self.assertEqual(output["components"]["u2"]["Name"], "ChipA123")

    def test_voltage_rating_is_not_capacitance(self):
        self.assertEqual(value_kind("16V"), "rating")
        self.assertEqual(value_kind("1u0"), "capacitance")
        self.assertEqual(value_kind("19M2"), "frequency")

    def test_reference_value_fusion_uses_independent_ocr_evidence(self):
        ds = [detection("resistor", [40,35,60,45])]
        ts = [token("R12100", [35,48,80,60], 1), token("R12", [35,48,55,60], 2)]
        output, _, _ = bind_components(ds, ts, 120, 100, CFG)
        self.assertEqual(output["components"]["R12"]["value"], "100")

    def test_partial_high_score_ocr_does_not_erase_unit_evidence(self):
        ds = [detection("capacitor", [40,35,50,55])]
        reading = token("0.1", [10,38,35,48], 2)
        reading["observations"] = [{"text": "0.1uF", "confidence": .98}]
        ts = [token("C1", [15,25,35,35], 1), reading]
        output, _, _ = bind_components(ds, ts, 120, 100, CFG)
        self.assertEqual(output["components"]["C1"]["value"], "0.1uF")

    def test_short_numeric_ref_suffix_is_not_invented_value(self):
        ds = [detection("resistor", [40,35,60,45])]
        ts = [token("R13", [35,48,57,60], 1), token("R13470", [35,48,80,60], 2)]
        output, _, _ = bind_components(ds, ts, 120, 100, CFG)
        self.assertIn("R13", output["components"])
        self.assertEqual(output["components"]["R13"]["value"], "470")

    def test_global_assignment_resolves_locally_close_candidates(self):
        costs=np.array([[2.,29.],[5.,3.]])
        result=choose_global(costs,90,3,[],"designator")
        self.assertEqual(result[0][0],0)
        self.assertEqual(result[1][0],1)
        self.assertGreater(result[0][2],3)

    def test_duplicate_boxes_do_not_compete_for_one_reference(self):
        ds=[detection("inductor",[40,45,80,60]),detection("inductor",[43,46,73,59])]
        ts=[token("L1",[48,15,68,27],1)]
        local_cfg={"class_mapping":{"inductor":"inductor"}}
        _,records,debug=bind_components(ds,ts,120,100,local_cfg)
        self.assertEqual(sum(r["designator"] is not None for r in records),1)
        self.assertTrue(debug["duplicate_detections"])

    def test_adjacent_parts_are_not_duplicate_boxes(self):
        ds=[detection("resistor",[40,40,65,52]),detection("resistor",[40,53,65,65])]
        self.assertEqual(duplicate_detections(ds),{})

    def test_value_covering_reference_is_rejected(self):
        ds=[detection("inductor",[50,30,80,42])]
        ts=[token("L1",[30,44,48,58],1),token("1330H",[30,44,90,58],2),token("3.3uH",[53,44,90,58],3)]
        _,records,_=bind_components(ds,ts,130,100,{"class_mapping":{"inductor":"inductor"}})
        self.assertEqual(records[0]["value"],"3.3uH")

    def test_case_style_only_ranks_observed_alternative(self):
        ds=[detection("ic",[40,50,100,100])]
        ts=[token("u2",[55,15,75,25],1)]
        ts += [token(f"R{i}",[300+i*30,20,315+i*30,30],i+2) for i in range(1,10)]
        cfg={"class_mapping":{"ic":"ic"}}
        output,_,_=bind_components(ds,ts,650,150,cfg)
        self.assertIn("u2",output["components"])
        upper=token("U2",[55,15,75,25],30)
        upper["confidence"]=.9
        ts[0]["confidence"]=.97
        output,_,_=bind_components(ds,ts+[upper],650,150,cfg)
        self.assertIn("U2",output["components"])

    def test_partial_ball_number_does_not_steal_resistor_value(self):
        ds=[detection("resistor",[50,35,75,47])]
        ts=[token("R1",[35,48,56,62],1),token("470",[58,48,85,62],2),
            token("B3",[25,62,42,76],3),token("3",[25,62,42,76],4)]
        ts[-1]["confidence"]=.75
        _,records,_=bind_components(ds,ts,120,100,CFG)
        self.assertEqual(records[0]["value"],"470")

    def test_crop_jitter_does_not_override_better_unit_reading(self):
        ds=[detection("inductor",[50,30,80,42])]
        ts=[token("L1",[30,44,48,58],1),token("3.3uH",[54,44,90,60],2),
            token("3.30H",[54,44,87,59],3)]
        ts[1]["confidence"]=.996
        ts[2]["confidence"]=.969
        _,records,_=bind_components(ds,ts,130,100,{"class_mapping":{"inductor":"inductor"}})
        self.assertEqual(records[0]["value"],"3.3uH")

    def test_numberless_capacitor_keeps_value_without_invented_reference(self):
        ds=[detection("capacitor",[40,35,50,55])]
        ts=[token("10nF",[54,36,85,48],1)]
        cfg={**CFG,"component_text":{"binding":{"allow_unreferenced_fields":True}}}
        _,records,debug=bind_components(ds,ts,120,100,cfg)
        self.assertIsNone(records[0]["designator"])
        self.assertEqual(records[0]["value"],"10nF")
        self.assertEqual(debug["counts"]["resolved_designators"],0)

    def test_centered_ic_model_can_be_inside_body_without_reference(self):
        ds=[detection("ic",[20,20,120,120])]
        ts=[token("TMP118",[45,65,95,80],1),token("GPIO123",[23,45,60,60],2)]
        cfg={**CFG,"component_text":{"binding":{"allow_unreferenced_fields":True}}}
        _,records,_=bind_components(ds,ts,150,150,cfg)
        self.assertEqual(records[0]["Name"],"TMP118")
        self.assertIsNone(records[0]["designator"])

    def test_common_switch_led_and_jumper_prefixes_are_consistent(self):
        for kind,ref in [("switch","S1"),("led","LED1"),("jumper","JP1")]:
            ds=[detection(kind,[40,40,60,60])]
            _,records,_=bind_components(ds,[token(ref,[38,15,65,30],1)],100,100,{"class_mapping":{kind:kind}})
            self.assertEqual(records[0]["designator"],ref)

    def test_value_space_is_preserved_when_unit_is_separated(self):
        ds=[detection("resistor",[40,35,50,60])]
        ts=[token("R1",[55,28,75,38],1),token("10 K",[55,42,90,55],2)]
        _,records,_=bind_components(ds,ts,120,100,CFG)
        self.assertEqual(records[0]["value"],"10 K")

    def test_repeated_references_and_symbol_prefixes_are_split(self):
        ds=[detection("resistor",[35,40,45,70]),detection("resistor",[70,40,80,70])]
        ts=[token(">R23>R24",[40,20,110,35],1)]
        output,_,_=bind_components(ds,ts,150,100,CFG)
        self.assertIn("R23",output["components"])
        self.assertIn("R24",output["components"])

    def test_ambiguous_zero_in_unit_is_not_silently_rewritten(self):
        ds=[detection("inductor",[40,40,75,55])]
        ts=[token("L1",[45,15,65,28],1),token("4700hm",[40,60,90,75],2)]
        _,records,debug=bind_components(ds,ts,130,100,{"class_mapping":{"inductor":"inductor"}})
        self.assertIsNone(records[0]["value"])
        self.assertEqual(debug["ambiguous_units"][0]["raw_text"],"4700hm")

    def test_led_reference_fragment_does_not_become_model_name(self):
        for reference in ['LED1','LED3']:
            reading=token(reference,[20,45,47,57],1)
            reading['observations']=[{'text':reference[1:],'confidence':.98,'view':'tile','bbox':[23,45,47,57]}]
            output,records,debug=bind_components([detection('led',[50,40,70,60])],[reading],100,100,{'class_mapping':{'led':'led'}})
            self.assertEqual(output['components'][reference]['Name'],reference)
            self.assertEqual(records[0]['fields']['Name']['name_source'],'observed_full_LED_reference_same_glyph')
            self.assertTrue(any(r['reason']=='truncated_LED_reference_is_not_separate_model' for r in debug['rejected_matches']))

    def test_distinct_led_model_is_kept_despite_reference_fragment(self):
        reference=token('LED1',[20,45,47,57],1)
        reference['observations']=[{'text':'ED1','confidence':.98,'view':'tile','bbox':[23,45,47,57]}]
        for model in ['GL0603UG01','NCD0805R1']:
            texts=[reference,token(model,[30,70,85,82],2)]
            output,_,_=bind_components([detection('led',[50,40,70,60])],texts,110,110,{'class_mapping':{'led':'led'}})
            self.assertEqual(output['components']['LED1']['Name'],model)

    def test_led_null_name_is_not_unconditionally_replaced_by_key(self):
        output,_,_=bind_components([detection('led',[50,40,70,60])],[token('LED1',[20,45,47,57],1)],100,100,{'class_mapping':{'led':'led'}})
        self.assertIsNone(output['components']['LED1']['Name'])

    def test_isolated_ED_fragment_does_not_create_LED_designator(self):
        output,records,_=bind_components([detection('led',[50,40,70,60])],[token('ED1',[20,45,47,57],1)],100,100,{'class_mapping':{'led':'led'}})
        self.assertIsNone(records[0]['designator'])
        self.assertNotIn('LED1',output['components'])

    def test_separate_ED_model_glyph_is_not_treated_as_truncated_reference(self):
        texts=[token('LED1',[20,45,47,57],1),token('ED1',[40,72,66,84],2)]
        output,_,_=bind_components([detection('led',[50,40,70,60])],texts,110,110,{'class_mapping':{'led':'led'}})
        self.assertEqual(output['components']['LED1']['Name'],'ED1')


if __name__ == "__main__":
    unittest.main()
