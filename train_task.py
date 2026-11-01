"""Prepare the frozen Indian Pines task model from official HyperSIGMA weights."""
import argparse
from pathlib import Path
import torch
from tritac import ROOT,seed_everything,device_from,indian_scene,load_checkpoint
from hypersigma_task import (add_hypersigma_args,resolve_hypersigma_config,fit_pca_projector,
                             train_hypersigma_official)

def main():
    p=argparse.ArgumentParser(description=__doc__);add_hypersigma_args(p)
    p.add_argument('--image',type=Path,default=ROOT/'data/indian_pines/Indian_pines_uint8.npy')
    p.add_argument('--gt',type=Path,default=ROOT/'data/indian_pines/Indian_pines_gt.npy')
    p.add_argument('--seed',type=int,default=3704)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--output',type=Path,default=Path('runs/task/hypersigma_finetuned_indian.pt'))
    a=p.parse_args()
    if not a.hypersigma_spat_ckpt or not a.hypersigma_spec_ckpt:
        p.error('Supply both official pretrained weights with --hypersigma-spat-ckpt and --hypersigma-spec-ckpt')
    if a.no_hypersigma_pca:p.error('This release uses the PCA task input')
    seed_everything(a.seed);device=device_from(a.device);cfg=resolve_hypersigma_config(a,'indian')
    image,train,val,_,_=indian_scene(a.image,a.gt,cfg,a.seed)
    projector=fit_pca_projector(image,cfg['pca_components'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    train_hypersigma_official(projector(image),train,val,16,a,cfg,device,a.output)
    ckpt=load_checkpoint(a.output)
    ckpt.update(projector=projector.state(),split_seed=a.seed,num_classes=16,model_size=a.hypersigma_model_size)
    torch.save(ckpt,a.output)

if __name__=='__main__':main()
