import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import lvae.models  # noqa: F401 - register model factories
from hsi_data import DATASETS, adapt_bands
from lvae.models.registry import get_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unlabeled single-scene RD finetuning for HSI compression."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="indian")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--model", type=str, default="auto")
    parser.add_argument("--src-channels", type=int, default=100)
    parser.add_argument("--target-bands", type=int, default=100)
    parser.add_argument("--band-adapter", choices=["pad_edge", "pad_zero", "interpolate"], default="pad_edge")
    parser.add_argument("--rd-lambda", type=float, default=None)
    parser.add_argument("--latent-scale", type=float, default=None)
    parser.add_argument("--rate-loss-unit", choices=["bppix", "bpppb"], default=None)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--samples-per-epoch", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--val-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=10587)
    parser.add_argument("--save-dir", type=str, default="runs/domain_adapt")
    parser.set_defaults(freeze_encoder=False)
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable train-time spatial augmentation: hflip, vflip, and 90-degree rotations.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return torch.device(name)


def load_hsi_image(path):
    array = np.load(path)
    if array.ndim != 3:
        raise ValueError(f"Expected HSI image HxWxC or CxHxW, got {array.shape}")
    # Prefer HWC when the first two dimensions are spatial, as in Indian/Pavia/KSC.
    if array.shape[0] > array.shape[-1] and array.shape[1] > array.shape[-1]:
        array = np.moveaxis(array, -1, 0)
    elif array.shape[0] == array.shape[1]:
        array = np.moveaxis(array, -1, 0)
    tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


class SingleHSICropDataset(Dataset):
    def __init__(self, image, crop_size, samples_per_epoch, seed, random_crop=True, augment=False):
        self.image = image
        self.crop_size = int(crop_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.random_crop = bool(random_crop)
        self.augment = bool(augment)
        self.rng = random.Random(seed)
        _, h, w = image.shape
        if h < self.crop_size or w < self.crop_size:
            raise ValueError(f"crop_size={self.crop_size} exceeds image shape={tuple(image.shape)}")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        del index
        _, h, w = self.image.shape
        if self.random_crop:
            top = self.rng.randrange(0, h - self.crop_size + 1)
            left = self.rng.randrange(0, w - self.crop_size + 1)
        else:
            top = (h - self.crop_size) // 2
            left = (w - self.crop_size) // 2
        patch = self.image[:, top:top + self.crop_size, left:left + self.crop_size]
        if self.augment:
            patch = self._augment(patch)
        return patch

    def _augment(self, patch):
        if self.rng.random() < 0.5:
            patch = torch.flip(patch, dims=(-1,))
        if self.rng.random() < 0.5:
            patch = torch.flip(patch, dims=(-2,))
        k = self.rng.randrange(4)
        if k:
            patch = torch.rot90(patch, k=k, dims=(-2, -1))
        return patch.contiguous()


def infer_checkpoint_config(ckpt, args):
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    rd_lambda = args.rd_lambda
    if rd_lambda is None:
        rd_lambda = ckpt_args.get("rd_lambda", ckpt.get("rd_lambda", 4096.0))
    latent_scale = args.latent_scale
    if latent_scale is None:
        latent_scale = ckpt.get("latent_scale", ckpt_args.get("latent_scale", 4.0))
    rate_loss_unit = args.rate_loss_unit
    if rate_loss_unit is None:
        rate_loss_unit = ckpt_args.get("rate_loss_unit", "bppix")
    variant = ckpt.get("variant", ckpt_args.get("variant", "base"))
    return float(rd_lambda), float(latent_scale), rate_loss_unit, variant


def make_model(args, ckpt, device):
    rd_lambda, latent_scale, rate_loss_unit, variant = infer_checkpoint_config(ckpt, args)
    if variant != "base":
        raise ValueError(f"Unsupported checkpoint variant={variant}.")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    model_name = args.model
    if model_name == "auto":
        model_name = ckpt_args.get("model", "qarv_base")
    kwargs = dict(
        src_channels=args.src_channels,
        rd_lambda=rd_lambda,
        use_hsi_adapters=False,
        rate_loss_unit=rate_loss_unit,
        rate_conditioning=False,
        latent_scale=latent_scale,
    )
    if model_name == "qarv_base":
        kwargs["hsi_mode"] = True
    model = get_model(model_name, **kwargs)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    if args.freeze_encoder:
        for name, param in model.named_parameters():
            param.requires_grad = not name.startswith("encoder.")
    model.to(device)
    info = {
        "variant": variant,
        "model": model_name,
        "rd_lambda": rd_lambda,
        "latent_scale": latent_scale,
        "rate_loss_unit": rate_loss_unit,
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "latent_z_dims": list(getattr(model, "latent_z_dims", [])),
    }
    return model, info


@torch.no_grad()
def run_eval(model, loader, device, steps):
    model.eval()
    metrics = {}
    count = 0
    for batch in loader:
        batch = batch.to(device)
        stats = model(batch)
        for key, value in stats.items():
            if key == "im_hat":
                continue
            if isinstance(value, torch.Tensor):
                value = float(value.detach().cpu().item())
            elif not isinstance(value, (int, float)):
                continue
            metrics[key] = metrics.get(key, 0.0) + float(value)
        count += 1
        if count >= steps:
            break
    return {key: value / max(count, 1) for key, value in metrics.items()}


def main():
    args = parse_args()
    if min(args.steps, args.val_steps, args.val_interval, args.batch_size) < 1:
        raise ValueError("Steps, intervals and batch size must be positive")
    if args.crop_size < 64 or args.crop_size % 64:
        raise ValueError("crop-size must be a multiple of 64")
    if args.crop_size % 64 != 0:
        raise ValueError("--crop-size must be a multiple of 64 for the QARV codec.")
    if args.src_channels != args.target_bands:
        raise ValueError("--src-channels must match --target-bands.")
    set_seed(args.seed)
    device = resolve_device(args.device)

    image_path = args.image or DATASETS[args.dataset]["image"]
    image = load_hsi_image(image_path)
    image = adapt_bands(image, args.target_bands, args.band_adapter)

    trainset = SingleHSICropDataset(
        image,
        args.crop_size,
        args.samples_per_epoch,
        args.seed,
        random_crop=True,
        augment=not args.no_augment,
    )
    valset = SingleHSICropDataset(
        image,
        args.crop_size,
        max(args.val_steps, 1),
        args.seed + 17,
        random_crop=False,
        augment=False,
    )
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    valloader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model, model_info = make_model(args, ckpt, device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    save_dir = Path(args.save_dir) / args.dataset / f"lambda{model_info['rd_lambda']:g}"
    save_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "args": vars(args),
        "checkpoint": args.checkpoint,
        "image_path": image_path,
        "image_shape": list(image.shape),
        "device": str(device),
        "model_info": model_info,
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    history = []
    best_psnr = -float("inf")
    iterator = iter(trainloader)
    initial_val = run_eval(model, valloader, device, args.val_steps)
    print(json.dumps({"phase": "initial_val", **initial_val}, ensure_ascii=False))

    for step in range(1, args.steps + 1):
        model.train()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(trainloader)
            batch = next(iterator)
        batch = batch.to(device)
        stats = model(batch)
        loss = stats["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=args.max_grad_norm,
        )
        optimizer.step()

        row = {
            key: (float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else value)
            for key, value in stats.items()
            if key != "im_hat"
        }
        row["step"] = step
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if step % args.val_interval == 0 or step == args.steps:
            val = run_eval(model, valloader, device, args.val_steps)
            val["step"] = step
            val["phase"] = "val"
            print(json.dumps(val, ensure_ascii=False))
            if val.get("psnr", -float("inf")) > best_psnr:
                best_psnr = val["psnr"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "step": step,
                        "val": val,
                        "args": vars(args),
                        "stage": "domain_adapt",
                        "variant": model_info["variant"],
                        "src_channels": args.src_channels,
                        "latent_scale": model_info["latent_scale"],
                        "rd_lambda": model_info["rd_lambda"],
                    },
                    save_dir / "best.pt",
                )
            torch.save(
                {
                    "model": model.state_dict(),
                    "step": step,
                    "optimizer": optimizer.state_dict(),
                    "val": val,
                    "args": vars(args),
                    "stage": "domain_adapt",
                    "variant": model_info["variant"],
                    "src_channels": args.src_channels,
                    "latent_scale": model_info["latent_scale"],
                    "rd_lambda": model_info["rd_lambda"],
                },
                save_dir / "last.pt",
            )

    final_val = run_eval(model, valloader, device, args.val_steps)
    torch.save(
        {
            "model": model.state_dict(),
            "step": args.steps,
            "val": final_val,
            "args": vars(args),
            "stage": "domain_adapt",
            "variant": model_info["variant"],
            "src_channels": args.src_channels,
            "latent_scale": model_info["latent_scale"],
            "rd_lambda": model_info["rd_lambda"],
        },
        save_dir / "final.pt",
    )
    result = {
        "save_dir": str(save_dir),
        "initial_val": initial_val,
        "final_val": final_val,
        "best_psnr": best_psnr,
        "history": history,
    }
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
