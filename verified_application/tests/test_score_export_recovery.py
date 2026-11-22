import copy,unittest
from pcb_parser.score_export_recovery import recover_export_contacts

def fixture():
    target={'components':{'U1':{'type':'box','Name':'TEST','value':None,'bbox':[10,70,30,90]},'R1':{'type':'r','Name':'R1','value':'10k','bbox':[60,70,70,90]}},'pins':{'R1':{'pin_1':{'pinname':'1','point':{'x':60,'y':80}}}},'nets':{'final_net':{'hyperGraph':'(R1.1)','edges':{'edge_1':[{'x':30,'y':80},{'x':60,'y':80}]}}}}
    pins={'size':[100,100],'pin_candidates':[dict(candidate_id='p1',owner='U1',owner_status='assigned',number='5',pinname=None,confidence=.99,point=[30,20]),dict(candidate_id='p2',owner='R1',owner_status='assigned',number='1',pinname='1',confidence=.99,point=[60,20])]}
    graph={'networks':[dict(network_id='raw_net',pin_candidate_ids=['p1','p2'])]}
    return target,pins,graph

class ScoreRecoveryTests(unittest.TestCase):
    def test_nameless_numbered_ic_recovers_network_without_changing_old_output(self):
        target,pins,graph=fixture();before=copy.deepcopy(target)
        out,audit=recover_export_contacts(target,pins,graph,'ic')
        self.assertEqual(out['pins']['U1']['pin_5']['pinname'],'')
        self.assertEqual(out['nets']['final_net']['hyperGraph'],'(R1.1,U1.5)')
        self.assertEqual(out['pins']['R1'],before['pins']['R1']);self.assertEqual(target,before)
        self.assertEqual(out['nets']['final_net']['edges'],before['nets']['final_net']['edges'])
        self.assertEqual(len(audit['added']),1)
    def test_passive_number_allocated_without_collision_and_not_for_ic(self):
        target,pins,graph=fixture();pins['pin_candidates'][0].update(owner='R1',number=None)
        out,audit=recover_export_contacts(target,pins,graph,'passive')
        self.assertIn('pin_2',out['pins']['R1']);self.assertEqual(out['nets']['final_net']['hyperGraph'],'(R1.1,R1.2)')
        pins['pin_candidates'][0]['owner']='U1'
        out,_=recover_export_contacts(target,pins,graph,'combined');self.assertEqual(out,target)
    def test_existing_correct_name_and_geometry_never_overwritten(self):
        target,pins,graph=fixture();target['pins']['U1']={'pin_5':{'pinname':'ORIGINAL','point':{'x':31,'y':81}}}
        out,_=recover_export_contacts(target,pins,graph,'combined');self.assertEqual(out,target)
    def test_ambiguous_source_net_does_not_merge_final_nets(self):
        target,pins,graph=fixture();target['nets']['other']={'hyperGraph':'(U1.6)','edges':{}}
        target['pins']['U1']={'pin_6':{'pinname':'X','point':{'x':30,'y':70}}}
        pins['pin_candidates'].append(dict(candidate_id='p3',owner='U1',owner_status='assigned',number='6',pinname='X',confidence=.99,point=[30,30]))
        graph['networks'][0]['pin_candidate_ids'].append('p3')
        out,audit=recover_export_contacts(target,pins,graph,'ic');self.assertEqual(out,target);self.assertTrue(audit['skipped'])
    def test_unsupported_no_anchor_and_low_confidence_rejected(self):
        target,pins,graph=fixture();pins['pin_candidates'][0]['confidence']=.2
        out,_=recover_export_contacts(target,pins,graph,'ic');self.assertEqual(out,target)
        pins['pin_candidates'][0]['confidence']=.99;graph['networks'][0]['pin_candidate_ids']=['p1']
        out,_=recover_export_contacts(target,pins,graph,'ic');self.assertEqual(out,target)
    def test_duplicate_new_number_prefers_named_supported_candidate(self):
        target,pins,graph=fixture();pins['pin_candidates'].append({**pins['pin_candidates'][0],'candidate_id':'p3','pinname':'VCC'})
        graph['networks'][0]['pin_candidate_ids'].append('p3')
        out,audit=recover_export_contacts(target,pins,graph,'ic');self.assertEqual(out['pins']['U1']['pin_5']['pinname'],'VCC');self.assertEqual(len(audit['added']),1)
    def test_candidate_in_multiple_raw_nets_rejected(self):
        target,pins,graph=fixture();graph['networks'].append(dict(network_id='other',pin_candidate_ids=['p1','p2']))
        out,_=recover_export_contacts(target,pins,graph,'ic');self.assertEqual(out,target)
    def test_duplicate_anchor_on_other_raw_net_cannot_create_join(self):
        target,pins,graph=fixture()
        pins['pin_candidates'].append({**pins['pin_candidates'][1],'candidate_id':'p3','point':[70,20]})
        graph['networks']=[dict(network_id='original',pin_candidate_ids=['p2']),dict(network_id='duplicate',pin_candidate_ids=['p1','p3'])]
        out,audit=recover_export_contacts(target,pins,graph,'ic')
        self.assertEqual(out,target);self.assertTrue(audit['skipped'])
    def test_disabled_profile_preserves_input(self):
        target,pins,graph=fixture();out,audit=recover_export_contacts(target,pins,graph,'none');self.assertEqual(out,target);self.assertEqual(audit['added'],[])
    def test_unknown_profile_is_error(self):
        with self.assertRaises(ValueError):recover_export_contacts(*fixture(),'combned')
