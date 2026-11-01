"""Reconstruction test over HySpecNet example patches (estimated entropy rate)."""
import argparse
import json
from pathlib import Path
import torch
from tritac import ROOT,device_from,load_codec,load_cube,reconstruct,image_metrics,save_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/hyspecnet11k/test')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'weights/stage1_hyspec_l4096.pt')
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--max-files',type=int)
    p.add_argument('--output',type=Path,default=Path('outputs/stage1.json'))
    a=p.parse_args();device=device_from(a.device)
    files=sorted(a.data.glob('*.npy'))
    if a.max_files is not None:
        if a.max_files<1:p.error('--max-files must be positive')
        files=files[:a.max_files]
    if not files:raise FileNotFoundError(f'No .npy files in {a.data}')
    model,config=load_codec(a.checkpoint,device)
    rows=[]
    for path in files:
        image=load_cube(path,config['src_channels'])
        rec,rate=reconstruct(model,image,device)
        row={'file':path.name,**image_metrics(image,rec),**rate}
        rows.append(row);print(json.dumps(row),flush=True)
    means={k:sum(row[k] for row in rows)/len(rows) for k in rows[0] if k!='file'}
    save_json(a.output,{'sample_count':len(rows),'rate_type':'entropy estimate; not file size','mean':means,'samples':rows})
    print(json.dumps(means))

if __name__=='__main__':main()
