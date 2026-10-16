"""Power export behavior at the platform boundary, independent of image models."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'submission/platform_adapter'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exporter = load('power_export', 'component_export.py')
adapter = load('power_adapter', 'adapter.py')


class PowerExportTests(unittest.TestCase):
    def fixture(self, typ='gnd'):
        payload = {'components': {'R1': {'type':'resistor','Name':None,'value':'10K','bbox':[10,20,30,40]}},
                   'pins': {'R1': {'pin_1':{'pinname':'1','point':{'x':10,'y':30}}}}, 'nets':{}}
        record = {'component_id':'UNRESOLVED_DET_0002','detection_index':1,'type':typ,
                  'designator':None,'Name':'GND' if typ=='gnd' else None,'value':None,
                  'bbox_top_left':[50,40,80,65],'duplicate_of_detection_index':None}
        records = {'records':[record], 'prediction_only':True, 'coordinate_origin':'top_left',
                   'size':[200,100], 'image_sha256':'image-a'}
        ocr = {'prediction_only':True,'size':[200,100],'image_sha256':'image-a','texts':[]}
        return payload, records, ocr

    def caption(self, text, view='global', box=None):
        return {'text':text,'view':view,'confidence':.999,'bbox':box or [55,70,78,80]}

    def test_undesignated_symbol_is_exported_without_inventing_pin_or_net(self):
        payload, records, ocr = self.fixture(); before=deepcopy((payload,records,ocr))
        result,audit=exporter.export_power_components(payload,records,ocr)
        self.assertIn('UNRESOLVED_DET_0002',result['components'])
        self.assertEqual(result['components']['UNRESOLVED_DET_0002']['value'],None)
        self.assertEqual(result['pins']['UNRESOLVED_DET_0002'],{})
        self.assertEqual(result['pins']['R1'],payload['pins']['R1'])
        self.assertEqual(result['nets'],payload['nets'])
        self.assertEqual(result['components']['R1'],payload['components']['R1'])
        self.assertEqual((payload,records,ocr),before)
        self.assertEqual(audit['added_count'],1)

    def test_supply_mapping_and_public_power_boxes(self):
        for typ,expected,dimensions in [('vcc','v',(10,10)),('gnd','gnd',(20,19))]:
            result,_=exporter.export_power_components(*self.fixture(typ))
            component=adapter.to_target(result)['components']['UNRESOLVED_DET_0002']
            self.assertEqual(component['type'],expected)
            b=component['bbox'];self.assertEqual((b[2]-b[0],b[3]-b[1]),dimensions)

    def test_caption_uses_distinct_views_and_preserves_exact_spelling(self):
        for typ,text in [('gnd','AGND'),('vcc','+3V3')]:
            payload,records,ocr=self.fixture(typ)
            ocr['texts']=[self.caption(text),self.caption(text,'tile_0_r0')]
            result,audit=exporter.export_power_components(payload,records,ocr)
            self.assertEqual(result['components']['UNRESOLVED_DET_0002']['value'],text)
            self.assertEqual(audit['value_counts'],{'observed':1})

    def test_repeated_same_view_does_not_supply_caption(self):
        payload,records,ocr=self.fixture()
        ocr['texts']=[self.caption('GND')]*3
        result,_=exporter.export_power_components(payload,records,ocr)
        self.assertIsNone(result['components']['UNRESOLVED_DET_0002']['value'])

    def test_conflicting_supported_spellings_stay_unknown(self):
        payload,records,ocr=self.fixture()
        ocr['texts']=[self.caption(text,view) for text in ['GND','AGND'] for view in ['global','tile']]
        result,_=exporter.export_power_components(payload,records,ocr)
        self.assertIsNone(result['components']['UNRESOLVED_DET_0002']['value'])

    def test_ic_internal_pin_label_is_not_power_caption(self):
        payload,records,ocr=self.fixture()
        records['records'].append({'component_id':'U1','type':'ic','detection_index':2,'bbox_top_left':[40,68,100,95]})
        ocr['texts']=[self.caption('GND',view) for view in ['global','tile']]
        result,_=exporter.export_power_components(payload,records,ocr)
        self.assertIsNone(result['components']['UNRESOLVED_DET_0002']['value'])

    def test_duplicate_detections_are_not_exported_twice(self):
        payload,records,ocr=self.fixture()
        duplicate={**records['records'][0],'component_id':'UNRESOLVED_DET_0003','detection_index':2,'duplicate_of_detection_index':1}
        records['records'].append(duplicate)
        result,audit=exporter.export_power_components(payload,records,ocr)
        self.assertEqual(audit['added_count'],1)
        self.assertNotIn(duplicate['component_id'],result['components'])

    def test_cross_image_caption_is_rejected(self):
        payload,records,ocr=self.fixture();ocr['image_sha256']='image-b'
        with self.assertRaisesRegex(ValueError,'different images'):
            exporter.export_power_components(payload,records,ocr)

    def test_unresolved_ordinary_component_is_not_given_an_identity(self):
        payload,records,ocr=self.fixture('capacitor')
        result,audit=exporter.export_power_components(payload,records,ocr)
        self.assertEqual(result,payload);self.assertEqual(audit['added_count'],0)

    def test_missing_field_evidence_does_not_change_recognized_values(self):
        payload,records,ocr=self.fixture('inductor')
        records['records'][0]['designator']='L1'
        ocr['texts']=[self.caption('4.7uH')]
        before=deepcopy((records,ocr))
        debug=exporter.missing_field_diagnostics(records,ocr)
        self.assertEqual(debug['entries'][0]['missing_fields'],['value'])
        self.assertEqual(debug['entries'][0]['nearby_OCR'][0]['text'],'4.7uH')
        self.assertEqual((records,ocr),before)
        self.assertEqual(debug['values_or_names_changed'],0)


if __name__=='__main__':
    unittest.main()
