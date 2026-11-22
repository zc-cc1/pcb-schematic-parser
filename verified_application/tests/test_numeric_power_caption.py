import unittest
import cv2,numpy as np

from pcb_parser.numeric_power_caption import numeric_voltage,read_numeric_supply,outward_wire


def fixture():
    rgb=np.full((128,128,3),255,np.uint8)
    cv2.line(rgb,(50,45),(50,65),(190,0,0),2)
    cv2.line(rgb,(50,55),(72,55),(190,0,0),2)
    cv2.line(rgb,(72,55),(110,55),(0,160,0),2)
    for y in (38,46,54):cv2.rectangle(rgb,(33,y),(37,y+4),(0,0,200),-1)
    return rgb,[40.,45.,60.,65.],[61.,55.]


def token(text='+5V',view='source',confidence=.99):
    return {'text':text,'view':view,'confidence':confidence,'bbox':[33,38,38,59]}


class NumericPowerCaptionTests(unittest.TestCase):
    def observe(self,reader,words=None,budget=8):
        rgb,box,point=fixture()
        return read_numeric_supply(rgb,box,point,words or [],reader=reader,remaining_budget=budget)

    def test_accepts_literal_numeric_voltage_labels(self):
        for text,expected in [('+5V','+5V'),('-12V','-12V'),('+3.3 V','+3.3V'),(' .5v ','.5V'),('−5V','-5V')]:
            self.assertEqual(numeric_voltage(text),expected)

    def test_does_not_guess_digits_sign_or_named_rail(self):
        for text in ('SV','VCC','R5','+5V/+3V','5V+','1e3V','+V','5','+-5V'):
            self.assertIsNone(numeric_voltage(text))

    def test_missing_whole_page_text_can_be_recovered_from_local_crop(self):
        def reader(rgb,box,scale,view,rotation=0):return [token(view=f'{view}_{scale}_{rotation}')]
        result=self.observe(reader)
        self.assertEqual(result['status'],'accepted');self.assertEqual(result['caption'],'+5V')
        self.assertEqual(result['calls'],3)

    def test_existing_one_view_requires_two_additional_reads(self):
        result=self.observe(lambda rgb,box,scale,view,rotation=0:[token(view=f'{view}_{scale}_{rotation}')],[token()])
        self.assertEqual(result['status'],'accepted');self.assertEqual(result['calls'],2)

    def test_vertical_caption_is_rotated_before_reading(self):
        rotations=[]
        def reader(rgb,box,scale,view,rotation=0):
            rotations.append(rotation)
            return [token(view=f'{view}_{scale}_{rotation}')]
        self.assertEqual(self.observe(reader)['status'],'accepted')
        self.assertEqual(rotations,[90,90,90])

    def test_repeated_same_view_does_not_manufacture_consensus(self):
        result=self.observe(lambda *args,**kwargs:[token(view='same_view')])
        self.assertEqual(result['status'],'withheld');self.assertEqual(result['calls'],8)

    def test_conflicting_signs_are_withheld(self):
        result=self.observe(lambda *args,**kwargs:[token('-5V','fresh')],[token('+5V')])
        self.assertEqual(result['reason'],'conflicting_voltage_captions')
        self.assertEqual(result['calls'],1)

    def test_conflicting_source_labels_are_not_resolved_by_guessing(self):
        result=self.observe(None,[token('+5V'),token('+3.3V','other')])
        self.assertEqual(result['reason'],'conflicting_voltage_captions');self.assertEqual(result['calls'],0)

    def test_low_confidence_ocr_cannot_create_a_supply(self):
        result=self.observe(lambda *args,**kwargs:[token(confidence=.89)],budget=3)
        self.assertEqual(result['status'],'withheld');self.assertEqual(result['calls'],3)

    def test_named_supply_remains_on_existing_rule_path(self):
        result=self.observe(None,[token('VCC')])
        self.assertEqual(result['reason'],'named_supply_handled_by_existing_rule')
        self.assertEqual(result['calls'],0)

    def test_no_reader_does_not_infer_voltage_from_geometry(self):
        result=self.observe(None)
        self.assertEqual(result['status'],'withheld');self.assertEqual(result['calls'],0)

    def test_budget_exhaustion_is_explicit(self):
        result=self.observe(lambda *args,**kwargs:[token()],budget=0)
        self.assertEqual(result['status'],'withheld');self.assertEqual(result['calls'],0)

    def test_plain_wire_without_supply_crossbar_is_rejected(self):
        rgb,box,point=fixture();rgb[44:67,48:53]=255
        cv2.line(rgb,(50,55),(72,55),(190,0,0),2)
        self.assertIsNone(outward_wire(rgb,box,point))

    def test_white_gap_prevents_recovery_despite_three_ocr_views(self):
        rgb,box,point=fixture();rgb[53:58,65:70]=255
        result=read_numeric_supply(rgb,box,point,[token(view=str(i)) for i in range(3)],reader=None,remaining_budget=0)
        self.assertEqual(result['status'],'withheld');self.assertEqual(result['calls'],0)

    def test_wrong_direction_never_traces_through_symbol(self):
        rgb,box,point=fixture();point=[39.,55.]
        self.assertIsNone(outward_wire(rgb,box,point))

    def test_existing_three_views_still_require_visible_blue_caption(self):
        rgb,box,point=fixture();rgb[34:60,30:40]=255
        result=read_numeric_supply(rgb,box,point,[token(view=str(i)) for i in range(3)],reader=None,remaining_budget=0)
        self.assertEqual(result['reason'],'no_isolated_blue_caption')


if __name__=='__main__':unittest.main()
