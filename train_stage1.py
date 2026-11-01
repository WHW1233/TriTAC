import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lvae.datasets import get_hsi_npy_dataset
import lvae.models  # noqa: F401 - registers model factories
from lvae.models.registry import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-1 fixed-rate HSI compression training.")
    parser.add_argument("--data", type=str, required=True, help="Directory containing unpacked .npy HSI patches.")
    parser.add_argument("--model", choices=["qarv_base"], default="qarv_base")
    parser.add_argument("--variant", choices=["base"], default="base",
                        help="Released model family.")
    parser.add_argument("--src-channels", type=int, default=100)
    parser.add_argument("--first-bands", type=int, default=100,
                        help="Use only the first N spectral bands from each .npy sample.")
    parser.add_argument("--latent-scale", type=float, default=4.0,
                        help="Multiplier for QARV latent channel counts. Released checkpoints use 4.0.")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--val-steps", type=int, default=2)
    parser.add_argument("--val-interval", type=int, default=100, help="Run validation every N training steps.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=10587)
    parser.add_argument("--max-files", type=int, default=None)
    parser.set_defaults(use_adapters=False)
    parser.set_defaults(adapter_axes='spectral,spatial,frequency')
    parser.set_defaults(adapter_factor=1.0)
    parser.add_argument("--rd-lambda", type=float, default=4096.0)
    parser.add_argument("--rate-loss-unit", choices=["bppix", "bpppb"], default="bppix",
                        help="Use total bits per spatial pixel or average bits per band-pixel as the rate term.")
    parser.set_defaults(rate_conditioning=False)
    parser.add_argument("--save-dir", type=str, default="runs/stage1")
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


def make_model(args):
    kwargs = dict(
        src_channels=args.src_channels,
        rd_lambda=args.rd_lambda,
        use_hsi_adapters=args.use_adapters,
        adapter_axes=tuple(x.strip() for x in args.adapter_axes.split(",") if x.strip()),
        adapter_factor=args.adapter_factor,
        rate_loss_unit=args.rate_loss_unit,
        rate_conditioning=args.rate_conditioning,
        latent_scale=args.latent_scale,
    )
    if args.model == "qarv_base":
        kwargs["hsi_mode"] = True
    return get_model(args.model, **kwargs)


def run_eval(model, loader, device, steps):
    model.eval()
    metrics = {}
    count = 0
    with torch.no_grad():
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
    if args.first_bands is not None and args.src_channels != args.first_bands:
        raise ValueError("--src-channels must match --first-bands for stage-1 training.")
    if min(args.steps, args.val_steps, args.val_interval, args.batch_size) < 1:
        raise ValueError("Steps, intervals and batch size must be positive")
    if args.crop_size < 64 or args.crop_size % 64:
        raise ValueError("crop-size must be a multiple of 64")
    set_seed(args.seed)
    device = resolve_device(args.device)

    trainset, valset = get_hsi_npy_dataset(
        args.data,
        crop_size=args.crop_size,
        max_files=args.max_files,
        first_bands=args.first_bands,
        seed=args.seed,
    )
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    valloader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    model = make_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Append lambda to save_dir so different lambda runs don't clash
    base_save_dir = Path(args.save_dir)
    save_dir = base_save_dir / args.variant / f"latent{args.latent_scale:g}x" / f"lambda{args.rd_lambda:g}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "args": vars(args),
        "device": str(device),
        "train_samples": len(trainset),
        "val_samples": len(valset),
        "save_dir": str(save_dir),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "stage": "stage1",
        "latent_z_dims": list(getattr(model, "latent_z_dims", [])),
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    history = []
    best_psnr = -float("inf")
    best_step = 0
    iterator = iter(trainloader)

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        row = {
            key: (float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else value)
            for key, value in stats.items()
            if key != "im_hat"
        }
        row["step"] = step
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        # ---------- periodic validation ----------
        if step % args.val_interval == 0 or step == args.steps:
            val_metrics = run_eval(model, valloader, device, args.val_steps)
            val_metrics["step"] = step
            val_metrics["phase"] = "val"
            print(json.dumps(val_metrics, ensure_ascii=False))

            # Track best by PSNR
            val_psnr = val_metrics.get("psnr", -float("inf"))
            val_loss = val_metrics.get("loss", float("inf"))
            is_best = val_psnr > best_psnr
            if is_best:
                best_psnr = val_psnr
                best_step = step
                torch.save(
                    {
                        "model": model.state_dict(),
                        "step": step,
                        "val": val_metrics,
                        "args": vars(args),
                        "stage": "stage1",
                        "variant": args.variant,
                        "src_channels": args.src_channels,
                        "latent_scale": args.latent_scale,
                    },
                    save_dir / "best.pt",
                )
                print(json.dumps({"best_saved": True, "step": step, "val_psnr": val_psnr}, ensure_ascii=False))

            # Always save latest checkpoint (resumable)
            torch.save(
                {
                    "model": model.state_dict(),
                    "step": step,
                    "optimizer": optimizer.state_dict(),
                    "val": val_metrics,
                    "args": vars(args),
                    "stage": "stage1",
                    "variant": args.variant,
                    "src_channels": args.src_channels,
                    "latent_scale": args.latent_scale,
                },
                save_dir / "last.pt",
            )

    # ---------- end of training ----------
    # Save full history
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final save
    torch.save(
        {
            "model": model.state_dict(),
            "step": args.steps,
            "args": vars(args),
            "stage": "stage1",
            "variant": args.variant,
            "src_channels": args.src_channels,
            "latent_scale": args.latent_scale,
        },
        save_dir / "final.pt",
    )

    result = {
        "save_dir": str(save_dir),
        "best_psnr": best_psnr,
        "best_step": best_step,
        "final_train": history[-1] if history else {},
    }
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
