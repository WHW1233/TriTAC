import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

from hsi_data import compute_metrics


DEFAULT_HYPERSIGMA_ROOT = str(Path(__file__).parent / "third_party/HyperSIGMA")
DEFAULT_SPAT_CKPT = None
DEFAULT_SPEC_CKPT = None
OFFICIAL_CONFIGS = {"indian": {"img_size":145,"patch_size":8,"pca_components":20,
    "train_num":10,"val_num":5,"task_batch_size":1,"overlap_size":0,"epochs":500}}


def official_config(dataset):
    if dataset not in OFFICIAL_CONFIGS:
        raise ValueError(f"No HyperSIGMA official-style config for dataset={dataset}")
    return dict(OFFICIAL_CONFIGS[dataset])


def add_hypersigma_args(parser):
    parser.add_argument("--hypersigma-root", type=str, default=DEFAULT_HYPERSIGMA_ROOT)
    parser.add_argument("--hypersigma-spat-ckpt", type=str, default=DEFAULT_SPAT_CKPT)
    parser.add_argument("--hypersigma-spec-ckpt", type=str, default=DEFAULT_SPEC_CKPT)
    parser.add_argument("--hypersigma-img-size", type=int, default=None)
    parser.add_argument("--hypersigma-patch-size", type=int, default=None)
    parser.add_argument("--hypersigma-pca-components", type=int, default=None)
    parser.add_argument("--hypersigma-overlap-size", type=int, default=None)
    parser.add_argument("--hypersigma-task-batch-size", type=int, default=None)
    parser.add_argument("--hypersigma-task-epochs", type=int, default=None)
    parser.add_argument("--hypersigma-task-lr", type=float, default=6e-5)
    parser.add_argument("--hypersigma-weight-decay", type=float, default=0.05)
    parser.add_argument("--hypersigma-layer-decay", type=float, default=0.9)
    parser.add_argument("--hypersigma-eval-interval", type=int, default=10)
    parser.add_argument("--hypersigma-model-size", choices=["base", "large", "huge"], default="base")
    parser.add_argument("--no-hypersigma-pca", action="store_true")


def resolve_hypersigma_config(args, dataset):
    cfg = official_config(dataset)
    if args.hypersigma_img_size is not None:
        cfg["img_size"] = args.hypersigma_img_size
    if args.hypersigma_patch_size is not None:
        cfg["patch_size"] = args.hypersigma_patch_size
    if args.hypersigma_pca_components is not None:
        cfg["pca_components"] = args.hypersigma_pca_components
    if args.no_hypersigma_pca:
        cfg["pca_components"] = 0
    if args.hypersigma_overlap_size is not None:
        cfg["overlap_size"] = args.hypersigma_overlap_size
    if args.hypersigma_task_batch_size is not None:
        cfg["task_batch_size"] = args.hypersigma_task_batch_size
    if args.hypersigma_task_epochs is not None:
        cfg["epochs"] = args.hypersigma_task_epochs
    return cfg


def _import_hypersigma(root, segmentation=True):
    cls_root = Path(root) / "ImageClassification"
    if not cls_root.is_dir():
        raise FileNotFoundError(f"Cannot find HyperSIGMA ImageClassification at {cls_root}")
    cls_root = str(cls_root)
    if cls_root not in sys.path:
        sys.path.insert(0, cls_root)
    if segmentation:
        from model.ss_fusion_seg import SSFusionFramework  # noqa: WPS433
    else:
        from model.ss_fusion_cls import SSFusionFramework  # noqa: WPS433
    return SSFusionFramework


def _checkpoint_model_state(path):
    if path is None:
        return {}
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def _prefixed_official_weights(state, prefix, branch):
    weights = {}
    for key, value in state.items():
        if "decoder" in key or "mask_token" in key:
            continue
        if branch == "spat":
            if "patch_embed.proj" in key or "spat_map" in key or "spat_output_maps" in key:
                continue
            if "pos_embed" in key:
                continue
        if branch == "spec":
            if "patch_embed.proj" in key or "spat_map" in key or "fpn1.0.weight" in key:
                continue
            if "patch_embed" in key:
                continue
        weights[prefix + key] = value
    return weights


class PCABandProjector(nn.Module):
    def __init__(self, mean, components, scale):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("components", torch.as_tensor(components, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.n_components = int(self.components.shape[0])

    def forward(self, image):
        squeeze = False
        if image.ndim == 3:
            image = image.unsqueeze(0)
            squeeze = True
        if image.ndim != 4:
            raise ValueError(f"Expected CHW or BCHW tensor, got {tuple(image.shape)}")
        b, c, h, w = image.shape
        x = image.permute(0, 2, 3, 1).reshape(-1, c)
        x = (x - self.mean.to(x.device)) @ self.components.to(x.device).t()
        x = x / self.scale.to(x.device).clamp_min(1e-12)
        x = x.reshape(b, h, w, self.n_components).permute(0, 3, 1, 2).contiguous()
        return x.squeeze(0) if squeeze else x

    def state(self):
        return {
            "mean": self.mean.detach().cpu(),
            "components": self.components.detach().cpu(),
            "scale": self.scale.detach().cpu(),
        }


def fit_pca_projector(image, n_components):
    if n_components is None or int(n_components) <= 0 or int(n_components) >= int(image.shape[0]):
        return None
    c, h, w = image.shape
    x = image.permute(1, 2, 0).reshape(-1, c).cpu().numpy()
    pca = PCA(n_components=int(n_components), whiten=True, random_state=3704)
    pca.fit(x)
    scale = np.sqrt(np.maximum(pca.explained_variance_, 1e-12))
    return PCABandProjector(pca.mean_, pca.components_, scale)


def load_pca_projector(state):
    if state is None:
        return None
    return PCABandProjector(state["mean"], state["components"], state["scale"])


def apply_task_projector(image, projector):
    if projector is None:
        return image
    return projector(image)


def make_fixed_number_splits(gt, train_num, val_num, seed):
    gen = torch.Generator().manual_seed(seed)
    train = torch.zeros_like(gt)
    val = torch.zeros_like(gt)
    test = torch.zeros_like(gt)
    info = {}
    for cls in torch.unique(gt):
        cls = int(cls.item())
        if cls == 0:
            continue
        coords = torch.nonzero(gt == cls, as_tuple=False)
        coords = coords[torch.randperm(coords.shape[0], generator=gen)]
        n = int(coords.shape[0])
        n_train = min(int(train_num), n)
        n_val = min(int(val_num), max(n - n_train, 0))
        n_test = max(n - n_train - n_val, 0)
        if n_test == 0 and n > 2:
            if n_val > 0:
                n_val -= 1
            elif n_train > 1:
                n_train -= 1
            n_test = n - n_train - n_val
        train_coords = coords[:n_train]
        val_coords = coords[n_train:n_train + n_val]
        test_coords = coords[n_train + n_val:n_train + n_val + n_test]
        train[train_coords[:, 0], train_coords[:, 1]] = cls
        if n_val:
            val[val_coords[:, 0], val_coords[:, 1]] = cls
        if n_test:
            test[test_coords[:, 0], test_coords[:, 1]] = cls
        info[cls] = {"train": n_train, "val": n_val, "test": n_test, "total": n}
    return train, val, test, info


def _grid_starts(length, tile, overlap):
    if length <= tile:
        return [0]
    stride = max(tile - overlap, 1)
    count = int(math.ceil((length - tile) / float(stride))) + 1
    return np.linspace(0, length - tile, count, endpoint=True).astype(int).tolist()


def _pad_chw(image, target_h, target_w, label=False):
    h, w = image.shape[-2:]
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    if pad_h == 0 and pad_w == 0:
        return image
    if label:
        return F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h), value=0).squeeze(0)
    mode = "reflect" if pad_h < h and pad_w < w else "replicate"
    return F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h), mode=mode).squeeze(0)


class SegTileDataset(Dataset):
    def __init__(self, image, labels, img_size, overlap_size=0, require_labels=True):
        self.image = image
        self.labels = labels
        self.img_size = int(img_size)
        self.overlap_size = int(overlap_size)
        _, h, w = image.shape
        padded_h = max(h, self.img_size)
        padded_w = max(w, self.img_size)
        self.image = _pad_chw(image, padded_h, padded_w, label=False)
        self.labels = _pad_chw(labels, padded_h, padded_w, label=True)
        _, h, w = self.image.shape
        ys = _grid_starts(h, self.img_size, self.overlap_size)
        xs = _grid_starts(w, self.img_size, self.overlap_size)
        self.tiles = []
        for y in ys:
            for x in xs:
                lab = self.labels[y:y + self.img_size, x:x + self.img_size]
                if require_labels and not bool((lab > 0).any()):
                    continue
                self.tiles.append((y, x))
        if not self.tiles:
            raise ValueError("No segmentation tiles contain labeled pixels.")

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        y, x = self.tiles[index]
        patch = self.image[:, y:y + self.img_size, x:x + self.img_size]
        labels = self.labels[y:y + self.img_size, x:x + self.img_size]
        return patch, labels.long()


class HyperSIGMASegmentation(nn.Module):
    def __init__(self, root, spat_ckpt, spec_ckpt, img_size, in_channels, patch_size, classes, model_size):
        super().__init__()
        self.img_size = int(img_size)
        self.in_channels = int(in_channels)
        self.classes = int(classes)
        SSFusionFramework = _import_hypersigma(root, segmentation=True)
        self.model = SSFusionFramework(
            img_size=self.img_size,
            in_channels=self.in_channels,
            patch_size=int(patch_size),
            classes=self.classes,
            model_size=model_size,
        )
        self.loaded_keys, self.candidate_keys = self._load_pretrained(spat_ckpt, spec_ckpt)

    def _load_pretrained(self, spat_ckpt, spec_ckpt):
        model_state = self.model.state_dict()
        spat_state = _prefixed_official_weights(
            _checkpoint_model_state(spat_ckpt), "spat_encoder.", "spat"
        )
        spec_state = _prefixed_official_weights(
            _checkpoint_model_state(spec_ckpt), "spec_encoder.", "spec"
        )
        candidates = {**spat_state, **spec_state}
        matched = {
            key: value
            for key, value in candidates.items()
            if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
        }
        model_state.update(matched)
        self.model.load_state_dict(model_state, strict=False)
        return len(matched), len(candidates)

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} bands, got {x.shape[1]}")
        b, _, h, w = x.shape
        logits = self.model(x)
        return logits.reshape(b, h, w, self.classes).permute(0, 3, 1, 2).contiguous()

    def forward_with_features(self, x, feature_layers=None):
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} bands, got {x.shape[1]}")
        feature_layers = set(feature_layers or ("dr1", "dr2", "dr3", "dr4", "fused"))
        m = self.model
        b, _, h, w = x.shape

        img_feature1, img_feature2, img_feature3, img_feature4 = m.spat_encoder(x)
        spec_feature = m.spec_encoder(x)[0]
        spec_feature_pooled = m.pool(spec_feature).view(b, -1)

        spec_weights1 = m.fc_spec1(spec_feature_pooled).view(b, -1, 1, 1)
        spec_weights2 = m.fc_spec2(spec_feature_pooled).view(b, -1, 1, 1)
        spec_weights3 = m.fc_spec3(spec_feature_pooled).view(b, -1, 1, 1)
        spec_weights4 = m.fc_spec4(spec_feature_pooled).view(b, -1, 1, 1)

        ss_feature1 = (1 + spec_weights1) * img_feature1
        ss_feature2 = (1 + spec_weights2) * img_feature2
        ss_feature3 = (1 + spec_weights3) * img_feature3
        ss_feature4 = (1 + spec_weights4) * img_feature4

        dr1 = m.DR1(ss_feature1)
        dr2 = m.DR2(ss_feature2)
        dr3 = m.DR3(ss_feature3)
        dr4 = m.DR4(ss_feature4)
        fused = m.conv(torch.cat((dr1, dr2, dr3, dr4), dim=1))
        logits_chw = m.cls(fused)
        logits = logits_chw.permute(0, 2, 3, 1).reshape(b, h * w, self.classes)
        logits = logits.reshape(b, h, w, self.classes).permute(0, 3, 1, 2).contiguous()

        all_features = {
            "dr1": dr1,
            "dr2": dr2,
            "dr3": dr3,
            "dr4": dr4,
            "fused": fused,
            "logits": logits,
            "spec": spec_feature,
        }
        return logits, {key: value for key, value in all_features.items() if key in feature_layers}


def make_hypersigma_segmentation(args, cfg, in_channels, num_classes, device):
    model = HyperSIGMASegmentation(
        root=args.hypersigma_root,
        spat_ckpt=args.hypersigma_spat_ckpt,
        spec_ckpt=args.hypersigma_spec_ckpt,
        img_size=cfg["img_size"],
        in_channels=in_channels,
        patch_size=cfg["patch_size"],
        classes=num_classes,
        model_size=args.hypersigma_model_size,
    )
    return model.to(device)


def _make_official_optimizer(model, args):
    try:
        _import_hypersigma(args.hypersigma_root, segmentation=True)
        from mmengine.optim import build_optim_wrapper  # noqa: WPS433
        from mmcv_custom import layer_decay_optimizer_constructor_vit  # noqa: F401,WPS433

        optim_wrapper = dict(
            optimizer=dict(
                type="AdamW",
                lr=args.hypersigma_task_lr,
                betas=(0.9, 0.999),
                weight_decay=args.hypersigma_weight_decay,
            ),
            constructor="LayerDecayOptimizerConstructor_ViT",
            paramwise_cfg=dict(num_layers=12, layer_decay_rate=args.hypersigma_layer_decay),
        )
        return build_optim_wrapper(model, optim_wrapper).optimizer
    except Exception as exc:
        print(json.dumps({"hypersigma_optimizer_fallback": str(exc)}))
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.hypersigma_task_lr,
            weight_decay=args.hypersigma_weight_decay,
        )


def segmentation_ce_loss(logits, labels):
    mask = labels > 0
    if not bool(mask.any()):
        return logits.sum() * 0.0
    logits = logits.permute(0, 2, 3, 1)[mask]
    target = labels[mask].long() - 1
    return F.cross_entropy(logits, target)


def segmentation_accuracy(logits, labels):
    mask = labels > 0
    if not bool(mask.any()):
        return 0.0, 0
    pred = logits.argmax(dim=1) + 1
    correct = int((pred[mask] == labels[mask]).sum().item())
    total = int(mask.sum().item())
    return correct / max(total, 1), total


@torch.no_grad()
def predict_hypersigma_segmentation(model, image, device, img_size, overlap_size=0, batch_size=1):
    model.eval()
    c, h0, w0 = image.shape
    tile = int(img_size)
    padded_h = max(h0, tile)
    padded_w = max(w0, tile)
    image = _pad_chw(image, padded_h, padded_w, label=False)
    _, h, w = image.shape
    ys = _grid_starts(h, tile, int(overlap_size))
    xs = _grid_starts(w, tile, int(overlap_size))
    pred_acc = None
    count = torch.zeros(1, h, w)
    batch = []
    coords = []
    for y in ys:
        for x in xs:
            batch.append(image[:, y:y + tile, x:x + tile])
            coords.append((y, x))
            if len(batch) >= batch_size:
                pred_acc = _flush_prediction_batch(model, batch, coords, pred_acc, count, device)
                batch, coords = [], []
    if batch:
        pred_acc = _flush_prediction_batch(model, batch, coords, pred_acc, count, device)
    pred = (pred_acc / count.clamp_min(1.0)).argmax(dim=0) + 1
    return pred[:h0, :w0].long()


def _flush_prediction_batch(model, batch, coords, pred_acc, count, device):
    x = torch.stack(batch).to(device)
    logits = model(x).detach().cpu()
    if pred_acc is None:
        pred_acc = torch.zeros(logits.shape[1], count.shape[1], count.shape[2])
    for logit, (y, x0) in zip(logits, coords):
        h, w = logit.shape[-2:]
        pred_acc[:, y:y + h, x0:x0 + w] += logit
        count[:, y:y + h, x0:x0 + w] += 1
    return pred_acc


def train_hypersigma_official(
    task_image,
    train_labels,
    val_labels,
    num_classes,
    args,
    cfg,
    device,
    save_path,
):
    model = make_hypersigma_segmentation(args, cfg, task_image.shape[0], num_classes, device)
    loader = DataLoader(
        SegTileDataset(
            task_image,
            train_labels,
            cfg["img_size"],
            overlap_size=cfg["overlap_size"],
            require_labels=True,
        ),
        batch_size=cfg["task_batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        SegTileDataset(
            task_image,
            val_labels,
            cfg["img_size"],
            overlap_size=cfg["overlap_size"],
            require_labels=True,
        ),
        batch_size=cfg["task_batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    opt = _make_official_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"], eta_min=0)
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_pixels = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = segmentation_ce_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            acc, pixels = segmentation_accuracy(logits.detach(), y)
            total_loss += float(loss.detach().cpu())
            total_acc += acc * pixels
            total_pixels += pixels
        scheduler.step()
        if epoch == 1 or epoch % args.hypersigma_eval_interval == 0 or epoch == cfg["epochs"]:
            val_loss, val_acc = _evaluate_segmentation_loss(model, val_loader, device)
            pred = predict_hypersigma_segmentation(
                model,
                task_image,
                device,
                cfg["img_size"],
                cfg["overlap_size"],
                cfg["task_batch_size"],
            )
            metrics = compute_metrics(pred, val_labels, num_classes)
            row = {
                "task_epoch": epoch,
                "train_loss": total_loss / max(len(loader), 1),
                "train_oa": total_acc / max(total_pixels, 1),
                "val_loss": val_loss,
                "val_tile_oa": val_acc,
                "val": metrics,
                "lr": float(opt.param_groups[0]["lr"]),
            }
            print(json.dumps(row))
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(
                    {
                        "model": best_state,
                        "metrics": metrics,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "cfg": cfg,
                        "loaded_keys": model.loaded_keys,
                        "candidate_keys": model.candidate_keys,
                    },
                    save_path,
                )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _evaluate_segmentation_loss(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_pixels = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = segmentation_ce_loss(logits, y)
        acc, pixels = segmentation_accuracy(logits, y)
        total_loss += float(loss.detach().cpu())
        total_acc += acc * pixels
        total_pixels += pixels
    return total_loss / max(len(loader), 1), total_acc / max(total_pixels, 1)
