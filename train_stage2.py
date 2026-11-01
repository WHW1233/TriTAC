"""Train all three TriLM branches on both sides; codec/task backbones are frozen."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from hypersigma_task import SegTileDataset
from tritac import (ROOT,LAYERS,seed_everything,device_from,load_codec,load_task,
                    indian_scene,codec_forward,reconstruct,task_metrics,save_json,load_checkpoint)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage1-checkpoint',type=Path,default=ROOT/'weights/stage1_indian_l4096.pt')
    p.add_argument('--task-checkpoint',type=Path,default=ROOT/'weights/hypersigma_finetuned_indian.pt')
    p.add_argument('--image',type=Path,default=ROOT/'data/indian_pines/Indian_pines_uint8.npy')
    p.add_argument('--gt',type=Path,default=ROOT/'data/indian_pines/Indian_pines_gt.npy')
    p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--val-interval',type=int,default=500)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--task-lambda',type=float,default=1.0)
    p.add_argument('--recon-lambda',type=float,default=0.01,help='Value recorded in the distributed checkpoint.')
    p.add_argument('--seed',type=int,default=3704)
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--save-dir',type=Path,default=Path('runs/stage2'))
    a=p.parse_args()
    if min(a.steps,a.val_interval,a.batch_size)<1:p.error('steps/interval/batch-size must be positive')
    seed_everything(a.seed);device=device_from(a.device);a.save_dir.mkdir(parents=True,exist_ok=True)
    task,projector,cfg,split_seed=load_task(a.task_checkpoint,device)
    image,train,val,test,split=indian_scene(a.image,a.gt,cfg,split_seed)
    codec,config=load_codec(a.stage1_checkpoint,device,add_adapters=True)
    for name,param in codec.named_parameters():param.requires_grad=name.startswith('feature_adapters.')
    trainable=[p for p in codec.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(trainable,lr=a.lr)
    loader=DataLoader(SegTileDataset(image,train,cfg['img_size'],overlap_size=cfg['overlap_size'],require_labels=True),
                      batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed),num_workers=0)
    iterator=iter(loader)
    def validate():
        rec,_=reconstruct(codec,image,device)
        return task_metrics(task,projector,cfg,rec,val,device)
    def save(step,metrics):
        torch.save({'model':codec.state_dict(),'step':step,'stage':'stage2','src_channels':100,
                    'latent_scale':config['latent_scale'],'rd_lambda':config['rd_lambda'],
                    'args':{'model':'qarv_base',**{k:v for k,v in config.items() if k!='adapter_axes'}},
                    'training':{'task_lambda':a.task_lambda,'recon_lambda':a.recon_lambda,'split_seed':split_seed},
                    'val_metrics':metrics},a.save_dir/'best.pt')
    best=validate();save(0,best);history=[]
    for step in range(1,a.steps+1):
        codec.train()
        try:x,_=next(iterator)
        except StopIteration:iterator=iter(loader);x,_=next(iterator)
        x=x.to(device);stats=codec_forward(codec,x);rec=stats['im_hat']
        with torch.no_grad():_,teacher=task.forward_with_features(projector(x),LAYERS)
        _,student=task.forward_with_features(projector(rec),LAYERS)
        distill=torch.stack([F.mse_loss(student[k],teacher[k].detach()) for k in LAYERS]).mean()
        mse=F.mse_loss(rec,x)
        loss=stats['rate_loss']+a.task_lambda*distill+a.recon_lambda*mse
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable,2.0);optimizer.step()
        row={'step':step,'loss':loss.item(),'distill':distill.item(),'mse':mse.item()}
        history.append(row);print(json.dumps(row),flush=True)
        if step%a.val_interval==0 or step==a.steps:
            metrics=validate();print(json.dumps({'val':metrics}),flush=True)
            if metrics['oa']>=best['oa']:best=metrics;save(step,best)
    codec.load_state_dict(load_checkpoint(a.save_dir/'best.pt')['model'],strict=True)
    rec,_=reconstruct(codec,image,device)
    save_json(a.save_dir/'results.json',{'selected_val':best,'test':task_metrics(task,projector,cfg,rec,test,device),'history':history})

if __name__=='__main__':main()
