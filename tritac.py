"""Shared checkpoint loading, reconstruction and task evaluation."""
import io
import json
import math
from pathlib import Path
import random
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

import lvae.models
from lvae.models.registry import get_model
from lvae.datasets.hsi import _as_chw
from hsi_data import adapt_bands, compute_metrics, load_hsi

ROOT = Path(__file__).resolve().parent
LAYERS = ('dr1', 'dr2', 'dr3', 'dr4', 'fused')


def seed_everything(seed=3704):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return torch.device(name)


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def codec_config(checkpoint):
    args = checkpoint.get('args', {})
    info = checkpoint.get('codec_info', {})
    def value(key, default):
        return next((v for v in [info.get(key), checkpoint.get(key), args.get(key)] if v is not None), default)
    return dict(src_channels=int(value('src_channels',100)), latent_scale=float(value('latent_scale',4)),
                rd_lambda=float(value('rd_lambda',4096)), hsi_mode=True, rate_conditioning=False,
                rate_loss_unit='bppix', adapter_axes=('spectral','spatial','frequency'), adapter_factor=1.0)


def load_codec(path, device, add_adapters=False):
    checkpoint = load_checkpoint(path)
    state = checkpoint['model']
    has_adapters = any(k.startswith('feature_adapters.') for k in state)
    config = codec_config(checkpoint)
    model = get_model('qarv_base', use_hsi_adapters=has_adapters or add_adapters, **config)
    result = model.load_state_dict(state, strict=not (add_adapters and not has_adapters))
    if result.unexpected_keys or any(not k.startswith('feature_adapters.') for k in result.missing_keys):
        raise RuntimeError(f'Checkpoint mismatch: {result}')
    return model.to(device).eval(), config


def load_cube(path, bands=100):
    array = np.load(path, allow_pickle=False)
    array = _as_chw(array)
    tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
    if tensor.max() > 1:
        tensor = tensor / 255.0
    return adapt_bands(tensor.clamp(0,1), bands, 'pad_edge')


def pad_input(x):
    h,w=x.shape[-2:]
    ph,pw=(-h)%64,(-w)%64
    return F.pad(x,(0,pw,0,ph),mode='reflect' if ph<h and pw<w else 'replicate') if ph or pw else x


def codec_forward(model,x,detach=False):
    stats=model(pad_input(x),return_rec=True,detach_rec=detach)
    stats['im_hat']=stats['im_hat'][...,:x.shape[-2],:x.shape[-1]]
    return stats


def image_metrics(original,reconstruction):
    mse=F.mse_loss(reconstruction,original).item()
    dot=(reconstruction*original).sum(0)
    norm=reconstruction.square().sum(0).sqrt()*original.square().sum(0).sqrt()
    sam=torch.acos((dot/norm.clamp_min(1e-8)).clamp(-1,1)).mean().item()*180/math.pi
    return {'mse':mse,'psnr':-10*math.log10(max(mse,1e-12)),'sam_deg':sam}


@torch.no_grad()
def reconstruct(model,image,device,tile=128):
    if tile < 1:
        raise ValueError('tile must be positive')
    c,h,w=image.shape
    out=torch.empty_like(image)
    bits=0.0
    model.eval()
    for y in range(0,h,tile):
        for x in range(0,w,tile):
            patch=image[:,y:y+tile,x:x+tile].unsqueeze(0).to(device)
            padded=pad_input(patch)
            stats=model(padded,return_rec=True,detach_rec=True)
            out[:,y:y+tile,x:x+tile]=stats['im_hat'][0,:,:patch.shape[-2],:patch.shape[-1]].cpu()
            bits+=float(stats['bppix'])*padded.shape[-2]*padded.shape[-1]
    return out.clamp(0,1), {'estimated_bppix':bits/(h*w),'estimated_bpppb':bits/(h*w*c)}


@torch.no_grad()
def encode(model,image,device,depth=9,tile=128):
    """ZIP_STORED container: shape/coordinates plus native entropy-coded tiles."""
    if tile < 64 or tile % 64:
        raise ValueError('tile must be a positive multiple of 64')
    if not 1<=depth<=model.num_latents:
        raise ValueError('depth outside codec latent range')
    c,h,w=image.shape
    buffer=io.BytesIO()
    entries=[]
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as archive:
        for y in range(0,h,tile):
            for x in range(0,w,tile):
                patch=image[:,y:y+tile,x:x+tile]
                stream=model.compress(pad_input(patch.unsqueeze(0).to(device)),latent_depth=depth)
                name=f'tile_{len(entries)}.bin'
                archive.writestr(name,stream)
                entries.append({'name':name,'top':y,'left':x,'height':patch.shape[1],'width':patch.shape[2]})
        archive.writestr('metadata.json',json.dumps({'format':'TriTAC-tiles-v1','shape':[c,h,w],'depth':depth,'tiles':entries}))
    return buffer.getvalue()


@torch.no_grad()
def decode(model,stream):
    with zipfile.ZipFile(io.BytesIO(stream)) as archive:
        meta=json.loads(archive.read('metadata.json'))
        if meta['format']!='TriTAC-tiles-v1':raise ValueError('Unsupported container')
        out=torch.empty(meta['shape'],dtype=torch.float32)
        for entry in meta['tiles']:
            rec=model.decompress(archive.read(entry['name']))[0].cpu()
            y,x,h,w=(entry[k] for k in ['top','left','height','width'])
            out[:,y:y+h,x:x+w]=rec[:,:h,:w]
    return out.clamp(0,1)


def load_task(path,device):
    from hypersigma_task import HyperSIGMASegmentation, DEFAULT_HYPERSIGMA_ROOT, load_pca_projector
    ckpt=load_checkpoint(path)
    cfg=ckpt['cfg']
    model=HyperSIGMASegmentation(DEFAULT_HYPERSIGMA_ROOT,None,None,cfg['img_size'],cfg['pca_components'],
                                cfg['patch_size'],ckpt.get('num_classes',16),ckpt.get('model_size','base'))
    model.load_state_dict(ckpt['model'],strict=True)
    model.to(device).eval()
    for p in model.parameters():p.requires_grad=False
    projector=load_pca_projector(ckpt['projector']).to(device)
    return model,projector,cfg,ckpt.get('split_seed',3704)


@torch.no_grad()
def task_metrics(model,projector,cfg,image,labels,device):
    from hypersigma_task import predict_hypersigma_segmentation
    task_image=projector(image.to(device)).cpu()
    pred=predict_hypersigma_segmentation(model,task_image,device,cfg['img_size'],cfg['overlap_size'],cfg['task_batch_size'])
    return compute_metrics(pred,labels,16)


def indian_scene(image_path,gt_path,cfg,seed):
    from hypersigma_task import make_fixed_number_splits
    image,gt=load_hsi(image_path,gt_path)
    image=adapt_bands(image,100,'pad_edge')
    train,val,test,info=make_fixed_number_splits(gt,cfg['train_num'],cfg['val_num'],seed)
    return image,train,val,test,info


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
