"""Indian Pines classification using frozen codec and HyperSIGMA."""
import argparse
import json
from pathlib import Path
import torch
from tritac import (ROOT,device_from,load_codec,load_task,indian_scene,task_metrics,
                    reconstruct,image_metrics,save_json,seed_everything)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'weights/stage2_indian_l4096.pt')
    p.add_argument('--task-checkpoint',type=Path,default=ROOT/'weights/hypersigma_finetuned_indian.pt')
    p.add_argument('--image',type=Path,default=ROOT/'data/indian_pines/Indian_pines_uint8.npy')
    p.add_argument('--gt',type=Path,default=ROOT/'data/indian_pines/Indian_pines_gt.npy')
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--output',type=Path,default=Path('outputs/stage2.json'))
    a=p.parse_args();seed_everything();device=device_from(a.device)
    task,projector,cfg,seed=load_task(a.task_checkpoint,device)
    image,train,val,test,split=indian_scene(a.image,a.gt,cfg,seed)
    codec,_=load_codec(a.checkpoint,device)
    rec,rate=reconstruct(codec,image,device)
    results={'split_by_class':split,'clean':task_metrics(task,projector,cfg,image,test,device),
             'reconstructed':task_metrics(task,projector,cfg,rec,test,device),
             'reconstruction':image_metrics(image,rec),'rate':rate,'rate_type':'entropy estimate'}
    save_json(a.output,results);print(json.dumps(results,indent=2))

if __name__=='__main__':main()
