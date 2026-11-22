import unittest
from copy import deepcopy
import cv2
import numpy as np
from pcb_parser.wire_export_geometry import refine_wire_geometry,GeometryDraft


def fixture(segments):
    target={'components':{},'pins':{},'nets':{'wire':{'hyperGraph':'(A.1,B.2)',
        'edges':{str(i):[{'x':float(x),'y':float(y)} for x,y in edge] for i,edge in enumerate(segments)}}},
        'metadata':{'keep':True}}
    graph={'image_sha256':'image','size':[128,128],'coordinate_origin':'top_left','prediction_only':True,
           'nodes':[],'edges':[],'networks':[]}
    rgb=np.full((128,128,3),255,np.uint8)
    return target,rgb,graph


def line(rgb,a,b,color=(0,160,0)):
    cv2.line(rgb,(a[0],128-a[1]),(b[0],128-b[1]),color,1)


def run(data,**cfg):
    return refine_wire_geometry(*data,'image',cfg={'enabled':True,**cfg})


def turn_fixture():
    t,r,g=fixture([[[20,50],[100,80]]]);line(r,[20,50],[100,50]);line(r,[100,50],[100,80])
    g['nodes']=[{'node_id':'start','point':[20,78]},{'node_id':'end','point':[100,48]}]
    g['edges']=[{'edge_id':'e','source':'start','target':'end','polyline':[[20,78],[100,78],[100,48]]}]
    g['networks']=[{'network_id':'wire','edge_ids':['e'],'node_ids':['start','end']}]
    return t,r,g


class WireExportGeometryTests(unittest.TestCase):
    def test_orthogonal_corner_follows_two_continuous_observed_arms(self):
        data=fixture([[[15,100],[15,40]],[[15,40],[120,48]]])
        line(data[1],[15,100],[15,48])
        line(data[1],[15,48],[120,48])
        out,audit=run(data)
        self.assertTrue(any(e['rule']=='observed_orthogonal_corner' for e in audit['changes']))
        segments=list(out['nets']['wire']['edges'].values())
        self.assertTrue(all({'x':15.,'y':48.} in edge for edge in segments))
        self.assertEqual(run((out,data[1],data[2]))[0],out)

    def terminal_lead_fixture(self,*,kind='c',color=(190,0,0),length=18):
        target,rgb,graph=fixture([[[20,50],[20+length,50]],[[60,70],[100,70]]])
        target['components']={'T1':{'type':kind,'bbox':[10,44,18,56]},
                              'R2':{'type':'r','bbox':[55,65,65,75]}}
        target['pins']={'T1':{'pin_1':{'pinname':'1','point':{'x':20.,'y':50.}}},
                        'R2':{'pin_1':{'pinname':'1','point':{'x':60.,'y':70.}}}}
        target['nets']['wire']['hyperGraph']='(T1.1,R2.1)'
        line(rgb,[20,50],[20+length,50],color)
        line(rgb,[60,70],[100,70])
        return target,rgb,graph

    def test_red_non_resistive_terminal_lead_removed_but_other_edge_retained(self):
        data=self.terminal_lead_fixture()
        out,audit=run(data,suppress_red_terminal_leads=True)
        self.assertEqual(len(out['nets']['wire']['edges']),1)
        self.assertEqual(next(iter(out['nets']['wire']['edges'].values())),data[0]['nets']['wire']['edges']['1'])
        self.assertEqual(out['nets']['wire']['hyperGraph'],data[0]['nets']['wire']['hyperGraph'])
        self.assertTrue(any(e['rule']=='red_terminal_symbol_lead' for e in audit['changes']))

    def test_green_terminal_lead_is_retained(self):
        data=self.terminal_lead_fixture(color=(0,160,0))
        self.assertEqual(run(data,suppress_red_terminal_leads=True)[0],data[0])

    def test_red_lead_with_green_contact_at_endpoint_is_suppressed(self):
        data=self.terminal_lead_fixture()
        line(data[1],[20,50],[20,60])
        out,audit=run(data,suppress_red_terminal_leads=True)
        self.assertEqual(len(out['nets']['wire']['edges']),1)
        self.assertTrue(any(e['rule']=='red_terminal_symbol_lead' for e in audit['changes']))

    def test_red_lead_with_parallel_green_trace_is_retained(self):
        data=self.terminal_lead_fixture()
        line(data[1],[20,52],[38,52])
        self.assertEqual(run(data,suppress_red_terminal_leads=True)[0],data[0])

    def test_red_resistor_terminal_lead_is_retained(self):
        data=self.terminal_lead_fixture(kind='r')
        self.assertEqual(run(data,suppress_red_terminal_leads=True)[0],data[0])

    def test_long_red_terminal_lead_is_retained(self):
        data=self.terminal_lead_fixture(length=34)
        self.assertEqual(run(data,suppress_red_terminal_leads=True)[0],data[0])

    def ground_branch_fixture(self,*,pin_y=50,lead_color=(190,0,0)):
        target,rgb,graph=fixture([[[10,50],[43,50]],[[60,70],[100,70]]])
        target['components']={'R1':{'type':'r','bbox':[5,42,15,58]},
                              'R2':{'type':'r','bbox':[55,65,65,75]},
                              'GND1':{'type':'gnd','bbox':[47,44,58,56]}}
        target['pins']={'R1':{'pin_1':{'pinname':'1','point':{'x':20.,'y':float(pin_y)}}},
                        'R2':{'pin_1':{'pinname':'1','point':{'x':60.,'y':70.}}},
                        'GND1':{'pin_1':{'pinname':'GND','point':{'x':43.,'y':50.}}}}
        target['nets']['wire']['hyperGraph']='(R1.1,R2.1,GND1.1)'
        line(rgb,[10,50],[43,50],lead_color)
        line(rgb,[60,70],[100,70])
        return target,rgb,graph

    def test_red_ground_branch_with_pin_inside_segment_keeps_other_edges(self):
        data=self.ground_branch_fixture()
        out,audit=run(data,suppress_short_ground_stubs=True)
        self.assertEqual(len(out['nets']['wire']['edges']),1)
        self.assertEqual(next(iter(out['nets']['wire']['edges'].values())),data[0]['nets']['wire']['edges']['1'])
        self.assertEqual(out['nets']['wire']['hyperGraph'],data[0]['nets']['wire']['hyperGraph'])
        self.assertTrue(any(e['rule']=='red_ground_symbol_branch' for e in audit['changes']))

    def test_green_ground_branch_is_retained(self):
        data=self.ground_branch_fixture(lead_color=(0,160,0))
        self.assertEqual(run(data,suppress_short_ground_stubs=True)[0],data[0])

    def test_ground_branch_without_pin_on_segment_is_retained(self):
        data=self.ground_branch_fixture(pin_y=55)
        self.assertEqual(run(data,suppress_short_ground_stubs=True)[0],data[0])

    def ground_stub_fixture(self,length=23,*,ground_type='gnd',two_edges=False):
        target,rgb,graph=fixture([[[20,50],[20+length,50]]])
        target['components']={'R1':{'type':'r','bbox':[10,45,18,55]},
                             'GND1':{'type':ground_type,'bbox':[44,45,55,55]}}
        target['pins']={'R1':{'pin_1':{'pinname':'1','point':{'x':20.,'y':50.}}}}
        target['nets']['wire']['hyperGraph']='(R1.1,GND1.1)'
        if two_edges:
            target['nets']['wire']['edges']['2']=[{'x':float(20+length),'y':50.},{'x':float(20+length+5),'y':50.}]
        return target,rgb,graph

    def test_short_pin_to_ground_symbol_stub_is_suppressed_with_membership_kept(self):
        data=self.ground_stub_fixture()
        out,audit=run(data,suppress_short_ground_stubs=True)
        self.assertEqual(out['nets']['wire']['edges'],{})
        self.assertEqual(out['nets']['wire']['hyperGraph'],data[0]['nets']['wire']['hyperGraph'])
        self.assertEqual(out['components'],data[0]['components'])
        self.assertEqual(out['pins'],data[0]['pins'])
        self.assertTrue(any(e['rule']=='short_ground_symbol_stub' for e in audit['changes']))

    def test_long_ground_connection_is_kept(self):
        data=self.ground_stub_fixture(34)
        self.assertEqual(run(data,suppress_short_ground_stubs=True)[0],data[0])

    def test_multi_segment_ground_network_is_kept(self):
        data=self.ground_stub_fixture(two_edges=True)
        self.assertEqual(run(data,suppress_short_ground_stubs=True)[0],data[0])

    def test_non_ground_symbol_pair_is_kept(self):
        data=self.ground_stub_fixture(ground_type='unresolved')
        self.assertEqual(run(data,suppress_short_ground_stubs=True)[0],data[0])

    def test_observed_red_junction_splits_long_run(self):
        data=fixture([[[20,60],[110,60]]]);line(data[1],[20,60],[110,60])
        cv2.circle(data[1],(65,68),5,(190,0,0),-1)
        out,audit=run(data)
        self.assertEqual(len(out['nets']['wire']['edges']),2)
        self.assertEqual(audit['changes'][0]['rule'],'observed_junction_splits')

    def test_green_crossing_has_no_red_anchor_split(self):
        data=fixture([[[20,60],[110,60]]]);line(data[1],[20,60],[110,60]);line(data[1],[65,20],[65,110])
        out,_=run(data);self.assertEqual(out,data[0])

    def test_red_node_with_only_one_green_arm_is_not_branch_anchor(self):
        data=fixture([[[20,60],[110,60]]]);line(data[1],[65,60],[110,60])
        cv2.circle(data[1],(65,68),5,(190,0,0),-1)
        self.assertEqual(run(data)[0],data[0])

    def test_off_path_red_node_is_not_projected_onto_nearby_wire(self):
        data=fixture([[[20,60],[110,60]]]);line(data[1],[20,60],[110,60]);line(data[1],[20,75],[110,75])
        cv2.circle(data[1],(65,53),5,(190,0,0),-1)
        self.assertEqual(run(data)[0],data[0])

    def test_collinear_degree_two_segments_merge(self):
        data=fixture([[[20,60],[65,60]],[[65,60],[110,60]]]);line(data[1],[20,60],[110,60])
        out,_=run(data);self.assertEqual(len(out['nets']['wire']['edges']),1)

    def test_real_red_node_is_retained_even_at_degree_two(self):
        data=fixture([[[20,60],[65,60]],[[65,60],[110,60]]]);line(data[1],[20,60],[110,60])
        cv2.circle(data[1],(65,68),5,(190,0,0),-1)
        self.assertEqual(run(data)[0],data[0])

    def test_small_right_angle_is_not_merged_into_long_diagonal(self):
        data=fixture([[[20,60],[100,60]],[[100,60],[100,64]]]);line(data[1],[20,60],[100,60]);line(data[1],[100,60],[100,64])
        self.assertEqual(run(data)[0],data[0])

    def test_degree_three_joint_is_not_collapsed(self):
        data=fixture([[[20,60],[65,60]],[[65,60],[110,60]],[[65,60],[65,90]]])
        line(data[1],[20,60],[110,60]);line(data[1],[65,60],[65,90])
        self.assertEqual(run(data)[0],data[0])

    def test_plain_white_gap_is_not_bridged(self):
        data=fixture([[[20,60],[65,60]],[[65,60],[110,60]]]);line(data[1],[20,60],[110,60]);data[1][68,65]=255
        self.assertEqual(run(data)[0],data[0])

    def test_source_turn_is_recovered_from_supported_unique_polyline(self):
        data=turn_fixture();out,audit=run(data)
        self.assertEqual(len(out['nets']['wire']['edges']),2)
        self.assertTrue(any(e['rule']=='source_path_turns' for e in audit['changes']))

    def test_reversed_source_polyline_does_not_reverse_the_turn(self):
        data=turn_fixture();expected=run(data)[0]
        data[2]['edges'][0]['polyline'].reverse()
        self.assertEqual(run(data)[0],expected)

    def test_source_path_with_white_gap_cannot_replace_shortcut(self):
        data=turn_fixture();data[1][78,55]=255
        self.assertEqual(run(data)[0],data[0])

    def test_source_path_without_matching_net_identity_is_not_used(self):
        data=turn_fixture();data[2]['networks'][0]['network_id']='other'
        self.assertEqual(run(data)[0],data[0])

    def test_missing_source_edge_is_withheld(self):
        data=turn_fixture();data[2]['networks'][0]['edge_ids']=['missing']
        self.assertEqual(run(data)[0],data[0])

    def test_unobserved_source_path_cannot_override_pixels(self):
        data=turn_fixture();data[1][:]=255
        self.assertEqual(run(data)[0],data[0])

    def test_equal_length_alternate_routes_are_withheld(self):
        data=turn_fixture();line(data[1],[20,50],[20,80]);line(data[1],[20,80],[100,80])
        data[2]['edges'].append({'edge_id':'e2','source':'start','target':'end','polyline':[[20,78],[20,48],[100,48]]})
        data[2]['networks'][0]['edge_ids'].append('e2')
        self.assertEqual(run(data)[0],data[0])

    def test_source_path_cannot_cross_component_body(self):
        data=turn_fixture();data[0]['components']['body']={'type':'box','bbox':[50,45,75,55]}
        self.assertEqual(run(data)[0],data[0])

    def test_components_pins_metadata_and_memberships_are_immutable(self):
        data=turn_fixture();data[0]['components']['label']={'type':'box','bbox':[5,5,10,10]}
        data[0]['pins']['label']={'pin_1':{'pinname':'NC','point':{'x':5.,'y':5.}}}
        original=deepcopy(data[0]);out,_=run(data)
        self.assertEqual(data[0],original)
        for section in ('components','pins','metadata'):self.assertEqual(out[section],original[section])
        self.assertEqual(out['nets']['wire']['hyperGraph'],original['nets']['wire']['hyperGraph'])

    def test_repeated_execution_is_exactly_idempotent(self):
        data=turn_fixture();first,_=run(data);again,audit=run((first,data[1],data[2]))
        self.assertEqual(first,again);self.assertFalse(audit['changes'])

    def test_zero_budget_rolls_back_entire_draft(self):
        data=turn_fixture();out,audit=run(data,maximum_changes=0)
        self.assertEqual(out,data[0]);self.assertEqual(audit['status'],'withheld_budget')

    def test_disable_and_independent_turn_switch(self):
        data=turn_fixture();self.assertEqual(run(data,enabled=False)[0],data[0])
        self.assertEqual(run(data,source_path_turns=False)[0],data[0])

    def test_unknown_flags_and_invalid_budgets_are_rejected(self):
        for cfg in [{'unknown':True},{'enabled':'yes'},{'maximum_changes':-1}]:
            with self.assertRaises(ValueError):run(turn_fixture(),**cfg)

    def test_image_binding_mismatch_is_rejected(self):
        for key,value in [('image_sha256','other'),('coordinate_origin','bottom_left'),('size',[1,1]),('prediction_only',False)]:
            data=turn_fixture();data[2][key]=value
            with self.assertRaises(ValueError):run(data)

    def test_duplicate_segments_are_removed_only_in_same_raw_net(self):
        data=fixture([[[20,60],[110,60]],[[110,60],[20,60]]]);line(data[1],[20,60],[110,60])
        data[0]['nets']['other']=deepcopy(data[0]['nets']['wire']);out,_=run(data)
        self.assertEqual(len(out['nets']['wire']['edges']),1);self.assertEqual(len(out['nets']['other']['edges']),1)
        self.assertEqual(set(out['nets']),set(data[0]['nets']))

    def test_white_overpass_needs_model_separated_straight_pair(self):
        data=fixture([[[20,60],[65,60]],[[65,60],[110,60]]]);line(data[1],[20,60],[110,60]);line(data[1],[65,20],[65,110])
        data[1][66:71,63:68]=255
        self.assertEqual(run(data)[0],data[0])
        data[2]['nodes']=[{'node_id':'left','point':[20,68]},{'node_id':'right','point':[110,68]},
            {'node_id':'horizontal','point':[65,68],'kind':'unresolved_crossing_branch'}]
        data[2]['edges']=[{'edge_id':'a','source':'left','target':'horizontal','polyline':[[20,68],[65,68]]},
            {'edge_id':'b','source':'horizontal','target':'right','polyline':[[65,68],[110,68]]}]
        data[2]['networks']=[{'network_id':'wire','node_ids':['left','right','horizontal'],'edge_ids':['a','b']}]
        data[2]['edges'][0]['polyline'].reverse()
        out,_=run(data);self.assertEqual(len(out['nets']['wire']['edges']),1)


if __name__=='__main__':unittest.main()
