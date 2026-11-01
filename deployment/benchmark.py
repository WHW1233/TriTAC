"""Measure actual stream bytes and synchronized encode+decode latency."""
import argparse
import io
import json
from pathlib import Path
import statistics
import time
import zipfile
import torch
from tritac import (ROOT,device_from,load_codec,load_cube,encode,decode,image_metrics,
                    save_json,load_task,indian_scene,task_metrics)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,default=ROOT/'data/indian_pines/Indian_pines_uint8.npy')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'weights/stage2_indian_l4096.pt')
    p.add_argument('--depth',type=int,choices=range(1,10),default=8)
    p.add_argument('--repeat',type=int,default=3)
    p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--tile',type=int,default=128)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--classification',action='store_true',help='Evaluate the bundled Indian Pines scene only.')
    p.add_argument('--output',type=Path,default=Path('outputs/codec.json'))
    a=p.parse_args()
    if a.repeat<1 or a.warmup<0:p.error('repeat >= 1 and warmup >= 0 required')
    if a.classification and a.image.resolve()!=(ROOT/'data/indian_pines/Indian_pines_uint8.npy').resolve():
        p.error('--classification requires the bundled Indian Pines image')
    device=device_from(a.device);model,config=load_codec(a.checkpoint,device)
    image=load_cube(a.image,config['src_channels']);model.compress_mode(True)
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    times=[]
    for i in range(a.warmup+a.repeat):
        sync();start=time.perf_counter()
        stream=encode(model,image,device,a.depth,a.tile);rec=decode(model,stream)
        sync();elapsed=(time.perf_counter()-start)*1000
        if i>=a.warmup:times.append(elapsed)
    with zipfile.ZipFile(io.BytesIO(stream)) as z:
        native_bytes=sum(e.file_size for e in z.infolist() if e.filename.endswith('.bin'))
    result={'depth':a.depth,'device':str(device),'shape':list(image.shape),'tile':a.tile,
            'container_bytes':len(stream),'native_stream_bytes':native_bytes,
            'container_bpppb':8*len(stream)/image.numel(),'native_bpppb':8*native_bytes/image.numel(),
            'median_encode_decode_ms':statistics.median(times),'timings_ms':times,
            'reconstruction':image_metrics(image,rec)}
    if a.classification:
        task,projector,cfg,seed=load_task(ROOT/'weights/hypersigma_finetuned_indian.pt',device)
        _,_,_,labels,_=indian_scene(a.image,ROOT/'data/indian_pines/Indian_pines_gt.npy',cfg,seed)
        result['classification']=task_metrics(task,projector,cfg,rec,labels,device)
    save_json(a.output,result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
