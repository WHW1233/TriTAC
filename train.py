import os
import argparse
import math
import random
import sys
import time
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models import DCAE

torch.set_num_threads(8)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

def spectral_angle_mapper_loss(x, y, eps=1e-8):
    dot = torch.sum(x * y, dim=1)
    norm_x = torch.norm(x, p=2, dim=1)
    norm_y = torch.norm(y, p=2, dim=1)
    cos_theta = dot / (norm_x * norm_y + eps)
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    sam = torch.acos(cos_theta)
    return sam.mean()

class Hyspecnet11kDataset(Dataset):
    def __init__(self, root, transform=None, is_train=True, num_bands=200):
        self.root = Path(root)
        self.transform = transform
        self.is_train = is_train
        self.num_bands = num_bands
        
        data_dir = self.root / 'data' if (self.root / 'data').exists() else self.root
        self.image_paths = sorted(data_dir.rglob('*.npy'))
        assert len(self.image_paths) > 0, f'Found {len(self.image_paths)} images in {root}.'
        
        split_idx = int(len(self.image_paths) * 0.9)
        if self.is_train:
            self.image_paths = self.image_paths[:split_idx]
        else:
            self.image_paths = self.image_paths[split_idx:]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        impath = self.image_paths[index]
        img = np.load(impath).astype(np.float32)
        # Only use the first num_bands channels
        if self.num_bands and img.shape[0] > self.num_bands:
            img = img[:self.num_bands, :, :]
        img = img / (img.max() + 1e-8)
        img = torch.from_numpy(img)
        return img, 0

def configure_optimizers(net, args):
    parameters = {n for n, p in net.named_parameters() if not n.endswith(".quantiles") and p.requires_grad}
    aux_parameters = {n for n, p in net.named_parameters() if n.endswith(".quantiles") and p.requires_grad}
    params_dict = dict(net.named_parameters())
    optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)), lr=args.learning_rate)
    aux_optimizer = optim.Adam((params_dict[n] for n in sorted(aux_parameters)), lr=args.aux_learning_rate)
    return optimizer, aux_optimizer

def train_one_epoch(model, train_dataloader, optimizer, aux_optimizer, epoch, args, lr_scheduler):
    model.train()
    device = next(model.parameters()).device
    alpha = args.alpha

    for i, (d, _) in enumerate(train_dataloader):
        d = d.to(device)
        B, C, H, W = d.shape
        num_pixels = B * H * W
        
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        
        out_net = model(d)
        
        bpp_loss = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in out_net["likelihoods"].values()
        )
        bpppb_loss = bpp_loss / C
        
        x_hat = out_net["x_hat"]
        mse_loss = nn.MSELoss()(x_hat, d)
        sam_loss = spectral_angle_mapper_loss(x_hat, d)
        
        d_human = (1 - alpha) * mse_loss + alpha * sam_loss
        loss = args.lmbda * d_human + bpp_loss
        
        loss.backward()

        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm) 
        
        optimizer.step()
        
        aux_loss = model.module.aux_loss() if torch.cuda.device_count() > 1 else model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if (i+1) % 10 == 0:
            print(f"Train epoch {epoch}: [{i*len(d)}/{len(train_dataloader.dataset)}]\tLoss: {loss.item():.4f} | Bpppb: {bpppb_loss.item():.4f} | MSE: {mse_loss.item():.4f} | SAM: {sam_loss.item():.4f}")

def test_epoch(model, test_dataloader, epoch, args):
    model.eval()
    device = next(model.parameters()).device
    alpha = args.alpha

    loss_sum = bpppb_sum = mse_sum = sam_sum = psnr_sum = 0
    count = 0
    
    with torch.no_grad():
        for d, _ in test_dataloader:
            d = d.to(device)
            B, C, H, W = d.shape
            num_pixels = B * H * W
            
            out_net = model(d)
            
            bpp_loss = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in out_net["likelihoods"].values()
            )
            bpppb_loss = bpp_loss / C
            
            x_hat = out_net["x_hat"]
            mse_loss = nn.MSELoss()(x_hat, d)
            sam_loss = spectral_angle_mapper_loss(x_hat, d)
            psnr = 10 * math.log10(1.0 / (mse_loss.item() + 1e-8))
            
            d_human = (1 - alpha) * mse_loss + alpha * sam_loss
            loss = args.lmbda * d_human + bpp_loss
            
            loss_sum += loss.item()
            bpppb_sum += bpppb_loss.item()
            mse_sum += mse_loss.item()
            sam_sum += sam_loss.item()
            psnr_sum += psnr
            count += 1
            
    print(f"Test epoch {epoch}: Loss: {loss_sum/count:.4f} | Bpppb: {bpppb_sum/count:.4f} | MSE: {mse_sum/count:.4f} | SAM: {sam_sum/count:.4f} | PSNR: {psnr_sum/count:.4f}")
    return loss_sum/count

def main(argv):
    parser = argparse.ArgumentParser(description="TriTAC Main Compression Network Training")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Training dataset")
    parser.add_argument("--test_dataset", type=str, default="/media/nercms/EXTERNAL_USB/satellite_data/HSI_data/hyspecnet11k/test_data", help="Test dataset")
    parser.add_argument("-e", "--epochs", default=50, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument("--aux-learning-rate", default=1e-3, type=float)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-batch-size", type=int, default=2)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=8.0, help="Bit-rate distortion parameter")
    parser.add_argument("--alpha", type=float, default=0.1, help="SAM vs MSE weight")
    parser.add_argument("--clip_max_norm", default=1.0, type=float)
    parser.add_argument("--save_path", type=str, default="./hsi_runs/")
    parser.add_argument("--in_channels", type=int, default=200)
    parser.add_argument("--eval_freq", type=int, default=1, help="Evaluate every N epochs")
    args = parser.parse_args(argv)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_dataset = Hyspecnet11kDataset(args.dataset, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    test_dataset = Hyspecnet11kDataset(args.test_dataset, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)
    
    net = DCAE(in_channels=args.in_channels).to(device)
    optimizer, aux_optimizer = configure_optimizers(net, args)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [40], gamma=0.1, last_epoch=-1)
    os.makedirs(args.save_path, exist_ok=True)

    print(f"Starting Main Network Training on {device} with in_channels={args.in_channels}")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    best_loss = float("inf")
    for epoch in range(args.epochs):
        train_one_epoch(net, train_dataloader, optimizer, aux_optimizer, epoch, args, lr_scheduler)
        lr_scheduler.step()

        test_loss = float("inf")
        if (epoch + 1) % args.eval_freq == 0:
            test_loss = test_epoch(net, test_dataloader, epoch, args)

        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "state_dict": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "aux_optimizer": aux_optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "loss": test_loss,
        }, os.path.join(args.save_path, "checkpoint_last.pth.tar"))

        # Save best checkpoint
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "loss": test_loss,
            }, os.path.join(args.save_path, "checkpoint_best.pth.tar"))
            print(f"  >>> New best model saved (loss={best_loss:.4f})")

if __name__ == "__main__":
    main(sys.argv[1:])
