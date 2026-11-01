from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

DATASETS = {"indian": {"image": "data/indian_pines/Indian_pines_uint8.npy", "gt": "data/indian_pines/Indian_pines_gt.npy"}}

def load_hsi(image_path, gt_path):
    image = np.load(image_path)
    gt = np.load(gt_path).astype(np.int64)
    if image.ndim != 3:
        raise ValueError(f"Expected HSI image HxWxC or CxHxW, got {image.shape}")
    if image.shape[:2] == gt.shape:
        image = np.moveaxis(image, -1, 0)
    elif image.shape[1:] != gt.shape:
        raise ValueError(f"Image shape {image.shape} and GT shape {gt.shape} do not match.")
    image = torch.from_numpy(np.ascontiguousarray(image)).float()
    if image.max() > 1.0:
        image = image / 255.0
    return image.clamp(0.0, 1.0), torch.from_numpy(gt)


def adapt_bands(image, target_bands, mode):
    c, h, w = image.shape
    if c == target_bands:
        return image
    if mode == "interpolate":
        x = image.permute(1, 2, 0).reshape(-1, 1, c)
        x = F.interpolate(x, size=target_bands, mode="linear", align_corners=False)
        return x.reshape(h, w, target_bands).permute(2, 0, 1).contiguous()
    if c > target_bands:
        return image[:target_bands]
    pad = target_bands - c
    if mode == "pad_zero":
        extra = torch.zeros(pad, h, w, dtype=image.dtype)
    elif mode == "pad_edge":
        extra = image[-1:].repeat(pad, 1, 1)
    else:
        raise ValueError(mode)
    return torch.cat([image, extra], dim=0)


def compute_metrics(pred, gt, num_classes):
    mask = gt > 0
    pred = pred[mask]
    gt = gt[mask]
    total = int(gt.numel())
    correct = int((pred == gt).sum().item())
    oa = correct / max(total, 1)
    recalls = []
    ious = []
    f1s = []
    conf = torch.zeros(num_classes + 1, num_classes + 1, dtype=torch.float64)
    for t, p in zip(gt.view(-1), pred.view(-1)):
        ti = int(t.item())
        pi = int(p.item())
        if not (1 <= pi <= num_classes):
            pi = 0
        conf[ti, pi] += 1
    for i in range(num_classes):
        idx = i + 1
        tp = conf[idx, idx]
        fn = conf[idx, :].sum() - tp
        fp = conf[:, idx].sum() - tp
        recall = tp / (tp + fn + 1e-12)
        precision = tp / (tp + fp + 1e-12)
        iou = tp / (tp + fp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        recalls.append(float(recall))
        ious.append(float(iou))
        f1s.append(float(f1))
    pe = float((conf.sum(0) * conf.sum(1)).sum() / ((conf.sum() ** 2) + 1e-12))
    kappa = (oa - pe) / (1 - pe + 1e-12)
    return {
        "oa": oa,
        "aa": float(np.mean(recalls)),
        "miou": float(np.mean(ious)),
        "macro_f1": float(np.mean(f1s)),
        "kappa": float(kappa),
        "pixels": total,
    }

