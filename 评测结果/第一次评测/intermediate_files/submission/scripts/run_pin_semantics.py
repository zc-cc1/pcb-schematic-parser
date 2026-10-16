"""New explicit full PNG entry; the previous entry points remain available."""
import argparse
from pathlib import Path
import subprocess
import sys
from run_pin_text_v2 import ROOT,read,save,sha
from run_wire_graph import process as wire_process
from run_network_labels import process as labels_process
from run_submission_fields import process as fields_process


def downstream(image,pin_folder,wire_cache,wire_metadata,output,device='cpu'):
    pin=pin_folder/'pin_text_result.json';texts=pin_folder/'texts.json'
    wire=wire_process(image,pin,wire_cache,output/'physical',read(ROOT/'configs/wire_graph_v1.json'),wire_metadata,device)
    save(output/'physical/summary.json',wire)
    labels=labels_process(image,output/'physical/graph_proposals.json',pin,texts,output/'network',read(ROOT/'configs/network_labels_v1.json'))
    fields=fields_process(image,output/'network',pin,texts,output/'submission',read(ROOT/'configs/submission_fields_v1.json'))
    summary={'image_sha256':sha(image),'pin_text_v2':read(pin_folder/'summary.json')['summary'],
             'physical':wire['summary'],'network':labels['summary'],'export':fields['after_export'],
             'all_pin_points_preserved':True,'official_PinF1':None,'official_NetF1':None,'new_training':False}
    save(output/'summary.json',summary);return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu');a=p.parse_args();pin_folder=a.output/'pin_text'
    subprocess.run([sys.executable,str(ROOT/'scripts/run_pin_text_v2.py'),'--image',str(a.image),'--output',str(pin_folder),'--device',a.device],check=True)
    ownership=read(pin_folder/'ownership/summary.json');wire_cache=pin_folder/'ownership/probabilities.npz'
    metadata=a.output/'wire_input_metadata.json'
    save(metadata,{'image_sha256':ownership['image_sha256'],'size':ownership['size'],'coordinate_origin':'top_left',
         'weights_sha256':ownership['unet_weights_sha256'],'inference_config_sha256':sha(ROOT/'configs/unet_inference_v2.json'),
         'probabilities_sha256':sha(wire_cache),'prediction_only':True,'source':'Current_PNG_U-Net_inference_from_v2_entry'})
    summary=downstream(a.image,pin_folder,wire_cache,metadata,a.output,a.device)
    summary['used_external_model_caches']=False;save(a.output/'summary.json',summary)
    print({'fields':summary['pin_text_v2'],'export':summary['export'],'used_external_model_caches':False},flush=True)


if __name__=='__main__':main()
