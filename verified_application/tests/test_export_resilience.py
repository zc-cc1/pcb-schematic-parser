import copy,json,os,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from pcb_parser.wire_graph_artifacts import partial_submission


def fixture():
    records=[dict(component_id='U1',designator='U1',type='ic',Name='TEST',value=None,bbox_top_left=[10,10,30,30]),dict(component_id='R1',designator='R1',type='resistor',Name='R1',value='10k',bbox_top_left=[60,10,70,30])]
    pins={'size':[100,100],'pin_candidates':[dict(candidate_id='p1',owner='U1',owner_status='assigned',number='1',pinname='VCC',point=[30,20]),dict(candidate_id='p2',owner='R1',owner_status='assigned',number='1',pinname='1',point=[60,20])]}
    graph={'edges':[dict(edge_id='e1',polyline=[[30,20],[60,20]])],'gap_candidates':[],'networks':[dict(network_id='n1',pin_candidate_ids=['p1','p2'],unresolved_junction_ids=[],edge_ids=['e1'])]}
    return pins,graph,records

class ExportResilienceTests(unittest.TestCase):
    def test_bad_pin_does_not_discard_good_component_pin_net(self):
        pins,graph,records=fixture();pins['pin_candidates'][0]['point']=[101,20]
        out,cov=partial_submission(pins,graph,records)
        self.assertEqual(set(out['components']),{'U1','R1'})
        self.assertEqual(out['nets']['n1']['hyperGraph'],'(R1.1)')
        self.assertEqual(cov['pending_pins'][0]['reasons'],['invalid_pin_geometry'])
    def test_bad_segment_isolated_from_valid_segment(self):
        pins,graph,records=fixture();graph['edges'][0]['polyline'].append([float('nan'),20])
        out,_=partial_submission(pins,graph,records)
        self.assertEqual(len(out['nets']['n1']['edges']),1)
        json.dumps(out,allow_nan=False)
    def test_malformed_segment_and_component_isolated_before_conversion(self):
        for endpoint in ([60], [60,None], 'bad'):
            with self.subTest(endpoint=endpoint):
                pins,graph,records=fixture();graph['edges'][0]['polyline'].append(endpoint)
                out,cov=partial_submission(pins,graph,records)
                self.assertEqual(set(out['components']),{'U1','R1'})
                self.assertEqual(len(out['nets']['n1']['edges']),1)
                self.assertTrue(cov['invalid_source_objects'])
    def test_malformed_component_isolated_before_conversion(self):
        pins,graph,records=fixture()
        records.append(dict(component_id='broken',designator='broken',type='resistor',bbox_top_left=[None,0,10,20]))
        out,cov=partial_submission(pins,graph,records)
        self.assertEqual(set(out['components']),{'U1','R1'});self.assertTrue(cov['invalid_source_objects'])
    def test_clean_payload_is_unchanged_and_input_not_mutated(self):
        from pcb_parser.export_resilience import sanitize_export
        out,_=partial_submission(*fixture());before=copy.deepcopy(out)
        clean,audit=sanitize_export(out,(100,100))
        self.assertEqual(clean,before);self.assertEqual(out,before);self.assertEqual(audit['removed'],[])
    def test_bad_owner_and_reference_removed_without_bad_json(self):
        from pcb_parser.export_resilience import sanitize_export
        from adapter import validate_target
        out,_=partial_submission(*fixture());out['pins']['ghost']={'pin_1':{'pinname':'','point':{'x':0,'y':0}}}
        out['nets']['n1']['hyperGraph']='(U1.1,R1.1,ghost.1,U1.1)'
        clean,audit=sanitize_export(out);validate_target(clean)
        self.assertEqual(clean['nets']['n1']['hyperGraph'],'(U1.1,R1.1)');self.assertNotIn('ghost',clean['pins']);self.assertTrue(audit['removed'])
    def test_nonfinite_component_does_not_remove_other_component(self):
        from pcb_parser.export_resilience import sanitize_export
        out,_=partial_submission(*fixture());out['components']['U1']['bbox'][0]=float('nan')
        clean,_=sanitize_export(out)
        self.assertEqual(set(clean['components']),{'R1'});self.assertEqual(clean['nets']['n1']['hyperGraph'],'(R1.1)')
    def test_empty_current_run_does_not_read_old_final(self):
        from pcb_parser.export_resilience import recover_current_run
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)/'new';work.mkdir();old=Path(tmp)/'final_results';old.mkdir()
            (old/'old_Pred.json').write_text(json.dumps(partial_submission(*fixture())[0]))
            self.assertIsNone(recover_current_run(work,'target'))
    def test_current_run_stage_fallback_preserves_valid_partial(self):
        from pcb_parser.export_resilience import recover_current_run
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'p_stage23/export/engineering_submission.json';p.parent.mkdir(parents=True)
            p.write_text(json.dumps(partial_submission(*fixture())[0]))
            result=recover_current_run(Path(tmp),'target')
            self.assertEqual(set(result['components']),{'U1','R1'})

class AgentFailureTests(unittest.TestCase):
    def run_fixture(self,fail_copy=False,fail_names=False,fail_pipeline=False,old_final=False):
        import agent
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);folder=Path(self.tmp.name)
        payload=partial_submission(*fixture())[0]
        def pipeline(command,**kwargs):
            stage=Path(command[command.index('--output')+1]);ocr=stage/'texts.json'
            if not fail_pipeline:
                agent.write(stage/'export/engineering_submission.json',payload)
                agent.write(stage/'export/completed_component_records.json',{'records':[]})
                agent.write(ocr,{'texts':[]})
                agent.write(stage/'summary.json',{'image_sha256':'synthetic','original_S23_input_sha256':{str(ocr):'synthetic'}})
            return types.SimpleNamespace(returncode=1 if fail_pipeline else 0)
        import sys
        ce=types.ModuleType('component_export');ce.export_power_components=lambda paper,*args:(copy.deepcopy(paper),{});ce.missing_field_diagnostics=lambda *args:{}
        names=types.ModuleType('pcb_parser.ic_name_selection')
        def select(target,*args):
            if fail_names:raise RuntimeError('injected refinement failure')
            return target,{}
        names.select_ic_names=select
        final=folder/'output/final_results/example_Pred.json'
        if old_final:agent.write(final,payload)
        with patch.dict(sys.modules,{'component_export':ce,'pcb_parser.ic_name_selection':names}),patch.dict(os.environ,{'PCB_BBOX_CONVENTION':'0','PCB_WIRE_CLEANUP':'0'}),patch.object(agent.subprocess,'run',pipeline):
            if fail_copy:
                with patch.object(agent.shutil,'copy2',side_effect=OSError('injected log failure')):row=agent.run_case(folder/'example.png',folder/'output','cpu','project',False)
            else:row=agent.run_case(folder/'example.png',folder/'output','cpu','project',False)
        return row,json.loads(final.read_text())
    def test_log_failure_does_not_erase_written_prediction(self):
        row,out=self.run_fixture(fail_copy=True)
        self.assertEqual(set(out['components']),{'U1','R1'});self.assertTrue(row['success']);self.assertTrue(row['diagnostic_errors'])
    def test_failed_refinement_falls_back_to_current_prediction(self):
        row,out=self.run_fixture(fail_names=True)
        self.assertEqual(set(out['components']),{'U1','R1'});self.assertTrue(row['degraded'])
    def test_failed_pipeline_never_reuses_previous_output(self):
        row,out=self.run_fixture(fail_pipeline=True,old_final=True)
        self.assertEqual(out,{'components':{},'pins':{},'nets':{}});self.assertFalse(row['success'])
