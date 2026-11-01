from pathlib import Path
from collections import OrderedDict, defaultdict
import math
import struct
import torch
import torch.nn as nn
import torch.nn.functional as tnf

import lvae.utils.coding as coding
import lvae.models.common as common
import lvae.models.entropy_coding as entropy_coding


class VRLVBlockBase(nn.Module):
    """ Vriable-Rate Latent Variable Block
    """
    default_embedding_dim = 256
    def __init__(
        self,
        width,
        zdim,
        enc_key,
        enc_width,
        embed_dim=None,
        kernel_size=7,
        mlp_ratio=2,
        latent_adapter=None,
    ):
        super().__init__()
        self.in_channels  = width
        self.out_channels = width
        self.enc_key = enc_key

        block = common.ConvNeXtBlockAdaLN
        embed_dim = embed_dim or self.default_embedding_dim
        self.resnet_front = block(width,   embed_dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.resnet_end   = block(width,   embed_dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.posterior0 = block(enc_width, embed_dim, kernel_size=kernel_size)
        self.posterior1 = block(width,     embed_dim, kernel_size=kernel_size)
        self.posterior2 = block(width,     embed_dim, kernel_size=kernel_size)
        self.post_merge = common.conv_k1s1(width + enc_width, width)
        self.posterior  = common.conv_k3s1(width, zdim)
        self.z_proj     = common.conv_k1s1(zdim, width)
        self.prior      = common.conv_k1s1(width, zdim*2)

        self.discrete_gaussian = entropy_coding.DiscretizedGaussian()
        self.is_latent_block = True
        self.latent_adapter = latent_adapter

    def transform_prior(self, feature, lmb_embedding):
        """ prior p(z_i | z_<i)

        Args:
            feature (torch.Tensor): feature map
        """
        feature = self.resnet_front(feature, lmb_embedding)
        pm, plogv = self.prior(feature).chunk(2, dim=1)
        plogv = tnf.softplus(plogv + 2.3) - 2.3 # make logscale > -2.3
        pv = torch.exp(plogv)
        return feature, pm, pv

    def transform_posterior(self, feature, enc_feature, lmb_embedding):
        """ posterior q(z_i | z_<i, x)

        Args:
            feature     (torch.Tensor): feature map
            enc_feature (torch.Tensor): feature map
        """
        assert feature.shape[2:4] == enc_feature.shape[2:4]
        enc_feature = self.posterior0(enc_feature, lmb_embedding)
        feature = self.posterior1(feature, lmb_embedding)
        merged = torch.cat([feature, enc_feature], dim=1)
        merged = self.post_merge(merged)
        merged = self.posterior2(merged, lmb_embedding)
        qm = self.posterior(merged)
        return qm

    def transform_latent_mean(self, qm, task_id=None):
        additional = {}
        if self.latent_adapter is not None:
            qm = self.latent_adapter(qm, task_id=task_id)
        return qm, additional

    def fuse_feature_and_z(self, feature, z):
        # add the new information carried by z to the feature
        feature = feature + self.z_proj(z)
        return feature

    def forward(self, feature, lmb_embedding, enc_feature=None, mode='trainval',
                get_latent=False, latent=None, t=1.0, strings=None, task_id=None,
                encode_latent=True):
        """ a complicated forward function

        Args:
            feature     (torch.Tensor): feature map
            enc_feature (torch.Tensor): feature map
        """
        feature, pm, pv = self.transform_prior(feature, lmb_embedding)

        additional = dict() # used to pack all returned values
        if mode == 'trainval': # training or validation
            qm = self.transform_posterior(feature, enc_feature, lmb_embedding)
            qm, latent_stats = self.transform_latent_mean(qm, task_id=task_id)
            additional.update(latent_stats)
            if self.training: # if training, use additive uniform noise
                z = qm + torch.empty_like(qm).uniform_(-0.5, 0.5)
                log_prob = entropy_coding.gaussian_log_prob_mass(pm, pv, x=z, bin_size=1.0, prob_clamp=1e-6)
                kl = -1.0 * log_prob
            else: # if evaluation, use residual quantization
                z, probs = self.discrete_gaussian(qm, scales=pv, means=pm)
                kl = -1.0 * torch.log(probs)
            additional['kl'] = kl
        elif mode == 'sampling':
            if latent is None: # if z is not provided, sample it from the prior
                z = pm + pv * torch.randn_like(pm) * t + torch.empty_like(pm).uniform_(-0.5, 0.5) * t
            else: # if `z` is provided, directly use it.
                assert pm.shape == latent.shape
                z = latent
        elif mode == 'compress': # encode z into bits
            if encode_latent:
                qm = self.transform_posterior(feature, enc_feature, lmb_embedding)
                qm, latent_stats = self.transform_latent_mean(qm, task_id=task_id)
                additional.update(latent_stats)
                indexes = self.discrete_gaussian.build_indexes(pv)
                strings = self.discrete_gaussian.compress(qm, indexes, means=pm)
                z = self.discrete_gaussian.quantize(qm, mode='dequantize', means=pm)
                additional['strings'] = strings
            else:
                # A missing latent transmits a zero residual around the conditional
                # prior.  The decoder can reproduce this value exactly without
                # side information, including when later latent blocks are coded.
                z = pm
                additional['strings'] = None
        elif mode == 'decompress': # decode z from bits
            if strings is None:
                z = pm
            else:
                indexes = self.discrete_gaussian.build_indexes(pv)
                z = self.discrete_gaussian.decompress(strings, indexes, means=pm)
        else:
            raise ValueError(f'Unknown mode={mode}')

        feature = self.fuse_feature_and_z(feature, z)
        feature = self.resnet_end(feature, lmb_embedding)
        if get_latent:
            additional['z'] = z.detach()
        return feature, additional

    def update(self):
        self.discrete_gaussian.update()


class VRLVBlockSmall(VRLVBlockBase):
    default_embedding_dim = 256
    def __init__(self, width, zdim, enc_key, enc_width, embed_dim=None, **kwargs):
        super(VRLVBlockBase, self).__init__()
        self.in_channels  = width
        self.out_channels = width
        self.enc_key = enc_key

        block = common.ConvNeXtBlockAdaLN
        enc_width = enc_width or width
        concat_ch = (width * 2) if (enc_width is None) else (width + enc_width)
        self.resnet_front = block(width, embed_dim, **kwargs)
        self.resnet_end   = block(width, embed_dim, **kwargs)
        self.posterior2   = block(width, embed_dim, **kwargs)
        self.post_merge = common.conv_k1s1(concat_ch, width)
        self.posterior  = common.conv_k3s1(width, zdim)
        self.z_proj     = common.conv_k1s1(zdim, width)
        self.prior      = common.conv_k1s1(width, zdim*2)

        self.discrete_gaussian = entropy_coding.DiscretizedGaussian()
        self.is_latent_block = True

    def transform_posterior(self, feature, enc_feature, lmb_embedding):
        """ posterior q(z_i | z_<i, x)

        Args:
            feature     (torch.Tensor): feature map
            enc_feature (torch.Tensor): feature map
        """
        assert feature.shape[2:4] == enc_feature.shape[2:4]
        merged = torch.cat([feature, enc_feature], dim=1)
        merged = self.post_merge(merged)
        merged = self.posterior2(merged, lmb_embedding)
        qm = self.posterior(merged)
        return qm


def mse_loss(fake, real):
    assert fake.shape == real.shape
    return tnf.mse_loss(fake, real, reduction='none').mean(dim=(1,2,3))


class FixedLambdaLossyVAE(nn.Module):
    log2_e = math.log2(math.e)
    MAX_LMB = 8192
    SUBSET_STREAM_MAGIC = b'QHS1'
    SUBSET_STREAM_HEADER = '<4sf3HH'

    def __init__(self, config: dict):
        super().__init__()
        # feature extractor (bottom-up path)
        self.encoder = common.FeatureExtractorWithEmbedding(config.pop('enc_blocks'))
        # latent variable blocks (top-down path)
        self.dec_blocks = nn.ModuleList(config.pop('dec_blocks'))
        feature_adapters = config.pop('feature_adapters', None)
        if feature_adapters is None:
            feature_adapters = nn.ModuleDict({
                "encoder": nn.ModuleDict(),
                "decoder": nn.ModuleDict(),
            })
        self.feature_adapters = feature_adapters
        width = self.dec_blocks[0].in_channels
        self.bias = nn.Parameter(torch.zeros(1, width, 1, 1))

        self.num_latents = len([b for b in self.dec_blocks if getattr(b, 'is_latent_block', False)])
        # loss function, for computing reconstruction loss
        self.distortion_name = 'mse'
        self.distortion_func = mse_loss
        self.rate_loss_unit = config.get('rate_loss_unit', 'bppix')
        if self.rate_loss_unit not in ('bppix', 'bpppb'):
            raise ValueError(f"rate_loss_unit must be 'bppix' or 'bpppb', got {self.rate_loss_unit}")

        self._setup_lmb_embedding(config)

        self.im_shift = float(config['im_shift'])
        self.im_scale = float(config['im_scale'])
        self.max_stride = config['max_stride']

        self.register_buffer('_dummy', torch.zeros(1), persistent=False)
        self._dummy: torch.Tensor

        self.compressing = False
        # self._stats_log = dict()
        self._logging_images = config.get('log_images', [])
        self._flops_mode = False

    def _setup_lmb_embedding(self, config):
        self.default_lmb = float(config['rd_lambda'])
        self.rate_conditioning = bool(config.get('rate_conditioning', False))
        self.embedding_lmb = float(config.get('embedding_lmb', 1.0))
        self.lmb_embed_dim = config['lmb_embed_dim']
        self.lmb_embedding = nn.Sequential(
            nn.Linear(self.lmb_embed_dim[0], self.lmb_embed_dim[1]),
            nn.GELU(),
            nn.Linear(self.lmb_embed_dim[1], self.lmb_embed_dim[1]),
        )
        self._sin_period = config['sin_period']

    def preprocess_input(self, im: torch.Tensor):
        """ Shift and scale the input image

        Args:
            im (torch.Tensor): a batch of images, values should be between (0, 1)
        """
        assert (im.shape[2] % self.max_stride == 0) and (im.shape[3] % self.max_stride == 0)
        assert (im.dim() == 4) and (0 <= im.min() <= im.max() <= 1) and not im.requires_grad
        x = im.clone().add_(self.im_shift).mul_(self.im_scale)
        return x

    def process_output(self, x: torch.Tensor):
        """ scale the decoder output from range (-1, 1) to (0, 1)

        Args:
            x (torch.Tensor): network decoder output, values should be between (-1, 1)
        """
        assert not x.requires_grad
        return self.scale_output(x)

    def scale_output(self, x: torch.Tensor):
        return x.clamp(min=-1.0, max=1.0).mul(0.5).add(0.5)

    def preprocess_target(self, im: torch.Tensor):
        """ Shift and scale the image to make it reconstruction target

        Args:
            im (torch.Tensor): a batch of images, values should be between (0, 1)
        """
        assert (im.dim() == 4) and (0 <= im.min() <= im.max() <= 1) and not im.requires_grad
        x = im.clone().add_(-0.5).mul_(2.0)
        return x


    def expand_to_tensor(self, input_, n):
        assert isinstance(input_, (torch.Tensor, float, int)), f'{type(input_)=}'
        if isinstance(input_, torch.Tensor) and (input_.numel() == 1):
            input_ = input_.item()
        if isinstance(input_, (float, int)):
            input_ = torch.full(size=(n,), fill_value=float(input_), device=self._dummy.device)
        assert input_.shape == (n,), f'{input_=}, {input_.shape=}'
        return input_

    def _lmb_scaling(self, lmb: torch.Tensor):
        # p = 3.0
        # lmb_input = torch.pow(lmb / self.MAX_LMB, 1/p) * self._sin_period
        lmb_input = torch.log(lmb) * self._sin_period / math.log(self.MAX_LMB)
        return lmb_input

    def _get_lmb_embedding(self, lmb, n):
        if self.rate_conditioning:
            lmb = self.expand_to_tensor(lmb, n=n)
        else:
            lmb = self.expand_to_tensor(self.embedding_lmb, n=n)
        scaled = self._lmb_scaling(lmb)
        embedding = common.sinusoidal_embedding(scaled, dim=self.lmb_embed_dim[0],
                                                max_period=self._sin_period)
        embedding = self.lmb_embedding(embedding)
        return embedding

    def get_bias(self, bhw_repeat=(1,1,1)):
        nB, nH, nW = bhw_repeat
        feature = self.bias.expand(nB, -1, nH, nW)
        return feature

    def _apply_feature_adapter(self, branch, key, feature):
        adapters = self.feature_adapters[branch]
        if key in adapters:
            return adapters[key](feature)
        return feature

    def _adapt_encoder_features(self, enc_features):
        if not self.feature_adapters["encoder"]:
            return enc_features
        adapted = OrderedDict()
        for key, feature in enc_features.items():
            adapted[key] = self._apply_feature_adapter("encoder", key, feature)
        return adapted

    def _next_latent_key(self, block_index):
        for block in self.dec_blocks[block_index + 1:]:
            if getattr(block, 'is_latent_block', False):
                return block.enc_key
            if isinstance(block, common.CompresionStopFlag):
                return None
        return None

    def _adapt_decoder_feature_after_block(self, block_index, block, feature):
        key = block.enc_key
        if self._next_latent_key(block_index) == key:
            return feature
        return self._apply_feature_adapter("decoder", key, feature)

    def forward_end2end(self, im: torch.Tensor, lmb: torch.Tensor, mode='trainval',
                        get_latent=False, task_id=None, latent_mask=None):
        x = self.preprocess_input(im)
        # ================ get lambda embedding ================
        emb = self._get_lmb_embedding(lmb, n=im.shape[0])
        # ================ Forward pass ================
        _, enc_features = self.encoder(x, emb)
        enc_features = self._adapt_encoder_features(enc_features)
        nB, _, xH, xW = x.shape
        feature = self.get_bias(bhw_repeat=(nB, xH//self.max_stride, xW//self.max_stride))
        lv_block_results = [] # all latent variable block results
        latent_i = 0
        if mode == 'compress':
            latent_mask = self.normalize_latent_mask(latent_mask)
        for i, block in enumerate(self.dec_blocks):
            if getattr(block, 'is_latent_block', False):
                f_enc = enc_features[block.enc_key]
                feature, stats = block(
                    feature,
                    emb,
                    enc_feature=f_enc,
                    mode=mode,
                    get_latent=get_latent,
                    task_id=task_id,
                    encode_latent=True if latent_mask is None else latent_mask[latent_i],
                )
                lv_block_results.append(stats)
                feature = self._adapt_decoder_feature_after_block(i, block, feature)
                latent_i += 1
            elif getattr(block, 'requires_embedding', False):
                feature = block(feature, emb)
            elif isinstance(block, common.CompresionStopFlag) and (mode == 'compress'):
                # no need to execute remaining blocks when compressing
                return lv_block_results
            else:
                feature = block(feature)
        return feature, lv_block_results

    def normalize_latent_mask(self, latent_mask=None, latent_depth=None):
        """Return a validated coarse-to-fine boolean mask for latent blocks.

        ``latent_depth`` selects a hierarchical prefix. ``latent_mask`` accepts
        either an integer bit mask or a sequence with one boolean per block.
        Block zero is the coarsest (s64 for qarv_base).
        """
        if latent_mask is not None and latent_depth is not None:
            raise ValueError('Specify either latent_mask or latent_depth, not both.')
        if latent_depth is not None:
            depth = int(latent_depth)
            if not 0 <= depth <= self.num_latents:
                raise ValueError(
                    f'latent_depth must be in [0, {self.num_latents}], got {depth}'
                )
            return tuple(i < depth for i in range(self.num_latents))
        if latent_mask is None:
            return (True,) * self.num_latents
        if isinstance(latent_mask, int):
            if latent_mask < 0 or latent_mask >= (1 << self.num_latents):
                raise ValueError(
                    f'latent_mask integer must fit {self.num_latents} bits, got {latent_mask}'
                )
            return tuple(bool(latent_mask & (1 << i)) for i in range(self.num_latents))
        mask = tuple(bool(value) for value in latent_mask)
        if len(mask) != self.num_latents:
            raise ValueError(
                f'Expected {self.num_latents} latent mask entries, got {len(mask)}'
            )
        return mask

    def latent_mask_to_int(self, latent_mask):
        mask = self.normalize_latent_mask(latent_mask)
        return sum((1 << i) for i, enabled in enumerate(mask) if enabled)

    def inspect_bitstream(self, string):
        """Read stream metadata without entropy-decoding the payload."""
        if string.startswith(self.SUBSET_STREAM_MAGIC):
            header_size = struct.calcsize(self.SUBSET_STREAM_HEADER)
            if len(string) < header_size:
                raise ValueError('Truncated QHS1 bitstream header.')
            _, lmb, nB, nH, nW, mask_bits = struct.unpack(
                self.SUBSET_STREAM_HEADER, string[:header_size]
            )
            mask = self.normalize_latent_mask(mask_bits)
            strings = coding.unpack_byte_string(string[header_size:])
            if len(strings) != sum(mask):
                raise ValueError(
                    f'Bitstream has {len(strings)} payloads for a {sum(mask)}-latent mask.'
                )
            return {
                'format': 'QHS1',
                'lambda': float(lmb),
                'batch': int(nB),
                'latent_hw': (int(nH), int(nW)),
                'latent_mask': mask,
                'latent_mask_int': int(mask_bits),
                'payload_bytes': tuple(len(value) for value in strings),
                'stream_bytes': len(string),
            }

        # Legacy streams encode all latents and use the original native-endian
        # float/shape header.
        header_size = struct.calcsize('f') + struct.calcsize('3H')
        if len(string) < header_size:
            raise ValueError('Truncated legacy QARV bitstream header.')
        lmb = struct.unpack('f', string[:4])[0]
        nB, nH, nW = struct.unpack('3H', string[4:header_size])
        strings = coding.unpack_byte_string(string[header_size:])
        if len(strings) != self.num_latents:
            raise ValueError(
                f'Legacy stream has {len(strings)} payloads; expected {self.num_latents}.'
            )
        return {
            'format': 'legacy',
            'lambda': float(lmb),
            'batch': int(nB),
            'latent_hw': (int(nH), int(nW)),
            'latent_mask': (True,) * self.num_latents,
            'latent_mask_int': (1 << self.num_latents) - 1,
            'payload_bytes': tuple(len(value) for value in strings),
            'stream_bytes': len(string),
        }

    def forward(self, batch, lmb=None, return_rec=False, detach_rec=True):
        task_id = None
        if isinstance(batch, (tuple, list)):
            im = batch[0]
            if len(batch) > 1:
                task_id = batch[1]
        else:
            im = batch
        im = im.to(self._dummy.device)
        nB, imC, imH, imW = im.shape # batch, channel, height, width

        # ================ Forward pass ================
        if lmb is None:
            lmb = self.expand_to_tensor(self.default_lmb, n=im.shape[0])
        assert isinstance(lmb, torch.Tensor) and lmb.shape == (nB,)
        x_hat, stats_all = self.forward_end2end(im, lmb, task_id=task_id)

        # ================ Compute Loss ================
        # rate
        kl_divergences = [stat['kl'].sum(dim=(1, 2, 3)) for stat in stats_all]
        ndims = float(imC * imH * imW)
        kl = sum(kl_divergences) / ndims # nats per band-pixel
        bpppb = kl * self.log2_e
        bppix = bpppb * imC
        if self.rate_loss_unit == 'bppix':
            rate_loss = bppix
        else:
            rate_loss = bpppb
        # distortion
        x_target = self.preprocess_target(im)
        distortion = self.distortion_func(x_hat, x_target)
        # rate + distortion
        loss = rate_loss + lmb * distortion
        loss = loss.mean(0)

        stats = OrderedDict()
        stats['loss'] = loss
        stats['rate_loss'] = rate_loss.mean(0)
        stats['distortion_loss'] = distortion.mean(0)

        # ================ Logging ================
        with torch.no_grad():
            # for training print
            stats['bppix'] = bppix.mean(0).item()
            stats['bpppb'] = bpppb.mean(0).item()
            stats['rate_loss_unit'] = self.rate_loss_unit
            stats['rd_lambda'] = float(lmb.mean().detach().item())
            stats[self.distortion_name] = distortion.mean(0).item()
            im_hat = self.process_output(x_hat.detach())
            im_mse = tnf.mse_loss(im_hat, im, reduction='mean')
            psnr = -10 * math.log10(im_mse.item())
            stats['psnr'] = psnr
            eps = 1e-8
            dot = (im_hat * im).sum(dim=1)
            denom = im_hat.square().sum(dim=1).sqrt() * im.square().sum(dim=1).sqrt()
            sam = torch.acos((dot / denom.clamp_min(eps)).clamp(-1.0, 1.0))
            stats['sam_deg'] = float(sam.mean().mul(180.0 / math.pi).item())

        if return_rec:
            rec_source = x_hat.detach() if detach_rec else x_hat
            stats['im_hat'] = self.scale_output(rec_source)
        return stats







    def compress_mode(self, mode=True):
        if mode:
            for block in self.dec_blocks:
                if hasattr(block, 'update'):
                    block.update()
        self.compressing = mode

    @torch.no_grad()
    def compress(self, im, lmb=None, latent_mask=None, latent_depth=None):
        lmb = self.default_lmb if lmb is None else lmb
        mask = self.normalize_latent_mask(latent_mask, latent_depth=latent_depth)
        lv_block_results = self.forward_end2end(
            im, lmb=lmb, mode='compress', latent_mask=mask
        )
        assert len(lv_block_results) == self.num_latents
        assert im.shape[0] == 1, f'Right now only support a single image, got {im.shape=}'
        all_lv_strings = [
            res['strings'][0]
            for enabled, res in zip(mask, lv_block_results)
            if enabled
        ]
        string = coding.pack_byte_strings(all_lv_strings)
        # Encode lambda, latent spatial shape, and subset mask in a versioned
        # header. The mask bits follow the coarse-to-fine latent block order.
        nB, _, imH, imW = im.shape
        lmb_value = float(lmb.item()) if isinstance(lmb, torch.Tensor) else float(lmb)
        header = struct.pack(
            self.SUBSET_STREAM_HEADER,
            self.SUBSET_STREAM_MAGIC,
            lmb_value,
            nB,
            imH // self.max_stride,
            imW // self.max_stride,
            self.latent_mask_to_int(mask),
        )
        string = header + string
        return string

    @torch.no_grad()
    def decompress(self, string):
        metadata = self.inspect_bitstream(string)
        lmb = metadata['lambda']
        nB = metadata['batch']
        nH, nW = metadata['latent_hw']
        mask = metadata['latent_mask']
        if metadata['format'] == 'QHS1':
            header_size = struct.calcsize(self.SUBSET_STREAM_HEADER)
        else:
            header_size = struct.calcsize('f') + struct.calcsize('3H')
        all_lv_strings = coding.unpack_byte_string(string[header_size:])

        lmb = self.expand_to_tensor(lmb, n=nB)
        lmb_embedding = self._get_lmb_embedding(lmb, n=nB)

        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        str_i = 0
        latent_i = 0
        for bi, block in enumerate(self.dec_blocks):
            if getattr(block, 'is_latent_block', False):
                if mask[latent_i]:
                    strs_batch = [all_lv_strings[str_i]]
                    str_i += 1
                else:
                    strs_batch = None
                feature, _ = block(feature, lmb_embedding, mode='decompress', strings=strs_batch)
                feature = self._adapt_decoder_feature_after_block(bi, block, feature)
                latent_i += 1
            elif getattr(block, 'requires_embedding', False):
                feature = block(feature, lmb_embedding)
            else:
                feature = block(feature)
        assert str_i == len(all_lv_strings), f'str_i={str_i}, len={len(all_lv_strings)}'
        assert latent_i == self.num_latents
        im_hat = self.process_output(feature)
        return im_hat


