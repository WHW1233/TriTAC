"""Encode/decode one HSI cube with a fixed latent prefix."""
import argparse
from pathlib import Path
import numpy as np
from tritac import ROOT, device_from, load_codec, load_cube, encode, decode

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['encode','decode'])
    p.add_argument('input',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'weights/stage2_indian_l4096.pt')
    p.add_argument('--depth',type=int,choices=range(1,10),default=9)
    p.add_argument('--tile',type=int,default=128)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args();device=device_from(a.device)
    model,config=load_codec(a.checkpoint,device);model.compress_mode(True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.action=='encode':
        stream=encode(model,load_cube(a.input,config['src_channels']),device,a.depth,a.tile)
        a.output.write_bytes(stream);print(f'Encoded {len(stream)} bytes')
    else:
        rec=decode(model,a.input.read_bytes())
        with a.output.open('wb') as f:np.save(f,rec.numpy(),allow_pickle=False)
        print(f'Decoded CHW {tuple(rec.shape)}; float32 [0, 1]')

if __name__=='__main__':main()
