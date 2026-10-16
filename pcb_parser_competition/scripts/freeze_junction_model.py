"""Freeze the pilot checkpoint and recipe using calibration only."""
import shutil
from run_wire_graph import ROOT,read,save,sha


def main():
    training=ROOT/'runs/training/junction_classifier_v1'
    evaluation=read(training/'evaluation.json')
    calibration=evaluation['scores']['calibration']
    if not calibration['accepted_count'] or calibration['accepted_count']!=calibration['accepted_correct']:
        raise ValueError('Calibration has accepted errors or no accepted samples; retain v1 baseline')
    checkpoint=training/'best.pt'
    if sha(checkpoint)!=evaluation['checkpoint_sha256']:raise ValueError('Training checkpoint changed')
    output=ROOT/'weights/junction_classifier_v1.pt';shutil.copy2(checkpoint,output)
    cfg=read(ROOT/'configs/wire_graph_v1.json')
    cfg.update(version='wire-graph-v2-junction-pilot',junction_model_available=True,
               junction_checkpoint='weights/junction_classifier_v1.pt',junction_checkpoint_sha256=sha(output),
               junction_confidence_threshold=read(ROOT/'configs/junction_training_v1.json')['confidence_threshold'],
               status='development_pilot_learned_junctions_not_official_complete_nets')
    save(ROOT/'configs/wire_graph_v2.json',cfg)
    print({'checkpoint_sha256':sha(output),'calibration_gate':calibration,'threshold':cfg['junction_confidence_threshold']})


if __name__=='__main__':main()
