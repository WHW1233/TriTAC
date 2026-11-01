import math


from collections import OrderedDict


import torch


import torch.nn as nn


import torch.nn.functional as tnf


from torch.hub import load_state_dict_from_url


from lvae.models.registry import register_model


import lvae.models.common as common


import lvae.models.qarv.model as qarv


from lvae.models.qarv.hsi_modules import make_hsi_feature_adapter


def _as_tuple(value):
    if isinstance(value, str):
        return tuple(v.strip() for v in value.split(',') if v.strip())
    return tuple(value)


def _scale_z_dims(z_dims, latent_scale):
    scale = float(latent_scale)
    if scale <= 0:
        raise ValueError(f"latent_scale must be positive, got {latent_scale}")
    return [max(1, int(round(z * scale))) for z in z_dims]


def _lv_block(width, zdim, enc_key, enc_width, **kwargs):
    return qarv.VRLVBlockBase(
        width,
        zdim,
        enc_key=enc_key,
        enc_width=enc_width,
        latent_adapter=None,
        **kwargs,
    )


def _make_feature_adapters(encoder_specs, decoder_specs, adapter_axes, adapter_factor):
    axes = _as_tuple(adapter_axes)

    def _branch(specs):
        return nn.ModuleDict({
            key: make_hsi_feature_adapter(
                channels,
                axes=axes,
                adapter_factor=adapter_factor,
                zero_init=True,
            )
            for key, channels in specs.items()
        })

    return nn.ModuleDict({
        "encoder": _branch(encoder_specs),
        "decoder": _branch(decoder_specs),
    })


@register_model
def qarv_base(
    rd_lambda=2048,
    pretrained=False,
    src_channels=3,
    hsi_mode=False,
    use_hsi_adapters=False,
    adapter_axes=("spectral", "spatial", "frequency"),
    adapter_factor=1.0,
    rate_loss_unit=None,
    rate_conditioning=None,
    latent_scale=8.0,
):
    cfg = dict()

    if hsi_mode:
        cfg['im_shift'] = -0.5
        cfg['im_scale'] = 2.0
    else:
        # mean and std computed on imagenet
        cfg['im_shift'] = -0.4546259594901961
        cfg['im_scale'] = 3.67572653978347
    # maximum downsampling factor
    cfg['max_stride'] = 64
    # images used during training for logging
    cfg['log_images'] = [] if hsi_mode else ['collie64.png', 'gun128.png', 'motor256.png']

    cfg['rd_lambda'] = float(rd_lambda)
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64
    cfg['rate_loss_unit'] = rate_loss_unit or ('bppix' if hsi_mode else 'bpppb')
    cfg['rate_conditioning'] = bool(rate_conditioning) if rate_conditioning is not None else (not hsi_mode)
    # _emb_dim = cfg['lmb_embed_dim'][1]

    # model configuration
    ch = 128
    enc_dims = [192, ch*3, ch*4, ch*4, ch*4]

    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]

    im_channels = int(src_channels)
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], kernel_size=7) for _ in range(6)],
        res_block(enc_dims[0]),
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], kernel_size=7) for _ in range(6)],
        common.SetKey('enc_s8'),
        res_block(enc_dims[1]),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], kernel_size=5) for _ in range(6)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2]),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=3) for _ in range(4)],
        common.SetKey('enc_s32'),
        res_block(enc_dims[3]),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=1) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*4, ch*4, ch*3, ch*2, ch*1]
    z_dims = _scale_z_dims([32, 32, 96, 8], latent_scale)
    cfg['dec_blocks'] = [
        # 1x1
        *[_lv_block(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1],
                    kernel_size=1, mlp_ratio=4)
          for _ in range(1)],
        res_block(dec_dims[0], kernel_size=1, mlp_ratio=4),
        common.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        res_block(dec_dims[1], kernel_size=3, mlp_ratio=3),
        *[_lv_block(dec_dims[1], z_dims[1], enc_key='enc_s32', enc_width=enc_dims[-2],
                    kernel_size=3, mlp_ratio=3)
          for _ in range(2)],
        res_block(dec_dims[1], kernel_size=3, mlp_ratio=3),
        common.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        res_block(dec_dims[2], kernel_size=5, mlp_ratio=2),
        *[_lv_block(dec_dims[2], z_dims[2], enc_key='enc_s16', enc_width=enc_dims[-3],
                    kernel_size=5, mlp_ratio=2)
          for _ in range(3)],
        res_block(dec_dims[2], kernel_size=5, mlp_ratio=2),
        common.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        *[_lv_block(dec_dims[3], z_dims[3], enc_key='enc_s8', enc_width=enc_dims[-4],
                    kernel_size=7, mlp_ratio=1.75)
          for _ in range(3)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        common.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[res_block(dec_dims[4], kernel_size=7, mlp_ratio=1.5) for _ in range(8)],
        common.patch_upsample(dec_dims[4], im_channels, rate=4)
    ]
    if use_hsi_adapters:
        cfg["feature_adapters"] = _make_feature_adapters(
            {
                "enc_s8": enc_dims[1],
                "enc_s16": enc_dims[2],
                "enc_s32": enc_dims[3],
                "enc_s64": enc_dims[4],
            },
            {
                "enc_s8": dec_dims[3],
                "enc_s16": dec_dims[2],
                "enc_s32": dec_dims[1],
                "enc_s64": dec_dims[0],
            },
            adapter_axes,
            adapter_factor,
        )

    model = qarv.FixedLambdaLossyVAE(cfg)
    model.latent_scale = float(latent_scale)
    model.latent_z_dims = tuple(z_dims)
    model.latent_block_repeats = (1, 2, 3, 3)

    if pretrained is True:
        url = 'https://huggingface.co/duanzh0/my-model-weights/resolve/main/qarv_base-2022-dec-12.pt'
        msd = load_state_dict_from_url(url)['model']
        model.load_state_dict(msd)
    elif pretrained: # str or Path
        msd = torch.load(pretrained)['model']
        model.load_state_dict(msd)
    return model

