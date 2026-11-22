import unittest
from copy import deepcopy
import cv2
import numpy as np
from pcb_parser.observed_terminal_memberships import recover_terminal_memberships


class NamedSupplyMembershipTests(unittest.TestCase):
    def fixture(self):
        rgb=np.full((180,240,3),255,np.uint8)
        cv2.line(rgb,(61,42),(79,42),(190,0,0),1)
        cv2.line(rgb,(70,42),(70,61),(190,0,0),1)
        cv2.line(rgb,(70,62),(70,110),(0,150,0),1)
        target={'components':{'SUPPLY':{'type':'v','Name':None,'value':'VCC','bbox':[65,118,75,128]},
                              'U9':{'type':'box','Name':'U9','value':None,'bbox':[55,50,85,65]}},
                'pins':{'SUPPLY':{},'U9':{'pin_7':{'pinname':'POWER','point':{'x':70.,'y':70.}}}},
                'nets':{'bus':{'hyperGraph':'(U9.7)','edges':{'edge_1':[{'x':70.,'y':118.},{'x':70.,'y':70.}]}}}}
        binding={'image_sha256':'fixture','size':[240,180],'coordinate_origin':'top_left','prediction_only':True}
        records={**binding,'records':[{'component_id':'SUPPLY','type':'vcc','bbox_top_left':[55,36,85,62],'detection_confidence':.95}]}
        pins={**binding,'pin_candidates':[{'candidate_id':'source','owner':'SUPPLY','owner_status':'assigned','point':[70.,59.], 'confidence':.98}]}
        ocr={**binding,'texts':[{'text':'VCC','confidence':.99,'bbox':[55,18,85,33],'view':'global'}]}
        return [target,rgb,dict(binding),records,pins,ocr,'fixture']

    def run_case(self,args,**cfg):return recover_terminal_memberships(*args,cfg={'enabled':True,**cfg})

    def test_confirmed_supply_exports_observed_pin_and_joins_unique_wire(self):
        args=self.fixture();old=deepcopy(args[0]);out,audit=self.run_case(args)
        self.assertEqual(out['nets']['bus']['hyperGraph'],'(SUPPLY.1,U9.7)')
        self.assertEqual(out['pins']['SUPPLY']['pin_1']['point'],{'x':70.,'y':121.})
        self.assertEqual(out['components'],old['components'])
        self.assertEqual(out['nets']['bus']['edges'],old['nets']['bus']['edges'])
        self.assertEqual(out['pins']['U9'],old['pins']['U9']);self.assertEqual(args[0],old)
        self.assertEqual(audit['attachments'],1)
        again,_=self.run_case([out,*args[1:]]);self.assertEqual(again,out)

    def test_rotated_supply_uses_source_contact_without_page_coordinates(self):
        args=self.fixture();args[1]=args[1].transpose(1,0,2).copy()
        for c in args[0]['components'].values():
            l,b,r,t=c['bbox'];c['bbox']=[180-t,240-r,180-b,240-l]
        for ps in args[0]['pins'].values():
            for p in ps.values():x,y=p['point'].values();p['point']={'x':180-y,'y':240-x}
        for net in args[0]['nets'].values():
            for edge in net['edges'].values():
                for p in edge:x,y=p.values();p.update(x=180-y,y=240-x)
        for doc in args[2:6]:doc['size']=[180,240]
        r=args[3]['records'][0];a,b,c,d=r['bbox_top_left'];r['bbox_top_left']=[b,a,d,c]
        args[4]['pin_candidates'][0]['point']=[59.,70.]
        t=args[5]['texts'][0];a,b,c,d=t['bbox'];t['bbox']=[b,a,d,c]
        out,_=self.run_case(args);self.assertEqual(out['nets']['bus']['hyperGraph'],'(SUPPLY.1,U9.7)')

    def test_no_supply_name_no_export(self):
        args=self.fixture();args[0]['components']['SUPPLY']['value']=None
        self.assertEqual(self.run_case(args)[0],args[0])

    def test_caption_conflict_or_wrong_owner_is_withheld(self):
        for conflict in ['different_text','other_owner','inside_ic']:
            args=self.fixture()
            if conflict=='different_text':args[5]['texts'][0]['text']='VDD'
            elif conflict=='other_owner':args[3]['records'].append({'component_id':'OTHER','type':'vcc','bbox_top_left':[55,36,85,62]})
            else:args[0]['components']['U9']['bbox']=[50,140,90,165]
            self.assertEqual(self.run_case(args)[0],args[0],conflict)

    def test_uncertain_or_multiple_source_contacts_are_withheld(self):
        for kind in ['confidence','owner','duplicate','text_hypothesis']:
            args=self.fixture();p=args[4]['pin_candidates'][0]
            if kind=='confidence':p['confidence']=.6
            elif kind=='owner':p['owner_status']='ambiguous'
            elif kind=='duplicate':args[4]['pin_candidates'].append({**p,'candidate_id':'other','point':[71.,59.]})
            else:p['text_hypothesis_primary_candidate_id']='synthetic'
            self.assertEqual(self.run_case(args)[0],args[0],kind)

    def test_source_contact_outside_symbol_is_withheld(self):
        args=self.fixture();args[4]['pin_candidates'][0]['point']=[70.,80.]
        self.assertEqual(self.run_case(args)[0],args[0])

    def test_short_contact_inside_other_component_is_withheld(self):
        args=self.fixture();args[0]['components']['U9']['bbox']=[60,119,80,140]
        self.assertEqual(self.run_case(args)[0],args[0])

    def test_white_gap_or_missing_t_cap_cannot_export_contact(self):
        for patch in [(slice(62,69),slice(65,76)),(slice(40,45),slice(59,81)),
                      (slice(47,54),slice(65,76))]:
            args=self.fixture();args[1][patch]=255
            self.assertEqual(self.run_case(args)[0],args[0])

    def test_ambiguous_networks_never_leave_a_partial_pin(self):
        args=self.fixture();args[0]['nets']['other']=deepcopy(args[0]['nets']['bus'])
        args[0]['nets']['other']['hyperGraph']='(U9.8)'
        self.assertEqual(self.run_case(args)[0],args[0])

    def test_unmatched_wire_never_leaves_a_partial_pin(self):
        args=self.fixture();args[0]['nets']={}
        self.assertEqual(self.run_case(args)[0],args[0])

    def test_conflicting_named_supply_on_target_net_is_withheld(self):
        args=self.fixture();args[0]['components']['OTHER']={'type':'v','Name':None,'value':'VDD','bbox':[150,120,170,140]}
        args[0]['pins']['OTHER']={'pin_1':{'pinname':'1','point':{'x':160.,'y':120.}}}
        args[0]['nets']['bus']['hyperGraph']='(OTHER.1,U9.7)'
        self.assertEqual(self.run_case(args)[0],args[0])

    def test_perpendicular_crossing_needs_junction_dot(self):
        args=self.fixture();rgb=args[1]
        cv2.line(rgb,(30,66),(100,66),(0,150,0),1)
        args[0]['nets']['bus']['edges']={'edge_1':[{'x':30.,'y':114.},{'x':100.,'y':114.}]}
        old=deepcopy(args[0]);out,_=self.run_case(args)
        self.assertEqual(out,old)
        cv2.circle(rgb,(70,66),6,(190,0,0),-1)
        out,_=self.run_case(args);self.assertEqual(out['nets']['bus']['hyperGraph'],'(SUPPLY.1,U9.7)')

    def test_nearby_member_pin_cannot_replace_actual_wire_projection(self):
        args=self.fixture();rgb=args[1]
        args[0]['nets']['other']=deepcopy(args[0]['nets']['bus'])
        args[0]['nets']['other']['hyperGraph']='(X.1)'
        args[0]['components']['X']={'type':'c','Name':'X','value':None,'bbox':[150,50,170,70]}
        args[0]['pins']['X']={'pin_1':{'pinname':'1','point':{'x':160.,'y':70.}}}
        args[0]['pins']['U9']['pin_7']['point']={'x':70.,'y':118.}
        cv2.line(rgb,(30,66),(100,66),(0,150,0),1)
        args[0]['nets']['bus']['edges']={'edge_1':[{'x':30.,'y':114.},{'x':100.,'y':114.}]}
        out,_=self.run_case(args)
        self.assertEqual(out['nets']['bus']['hyperGraph'],'(U9.7)')
        self.assertEqual(out['nets']['other']['hyperGraph'],'(SUPPLY.1,X.1)')

    def test_existing_supply_pin_is_preserved(self):
        args=self.fixture();args[0]['pins']['SUPPLY']={'pin_5':{'pinname':'NC','point':{'x':20,'y':20}}}
        self.assertEqual(self.run_case(args)[0],args[0])

    def test_dangling_membership_without_pin_is_withheld(self):
        args=self.fixture();args[0]['nets']['bus']['hyperGraph']='(SUPPLY.2,U9.7)'
        self.assertEqual(self.run_case(args)[0],args[0])

    def test_zero_attachment_budget_rolls_back_contact_together_with_net(self):
        args=self.fixture();out,audit=self.run_case(args,maximum_attachments=0)
        self.assertEqual(out,args[0]);self.assertEqual(audit['status'],'withheld_budget')

    def test_disabled_and_stale_evidence(self):
        args=self.fixture();self.assertEqual(recover_terminal_memberships(*args)[0],args[0])
        args[4]['image_sha256']='stale'
        with self.assertRaisesRegex(ValueError,'evidence_differs_from_image'):self.run_case(args)


if __name__=='__main__':unittest.main()
