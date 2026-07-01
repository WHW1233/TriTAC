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

# ==========================================
# 1. Downstream Task Networks
# ==========================================

class Classifier(nn.Module):
    def __init__(self, in_channels=202, num_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 64x64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 32x32
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=202, num_classes=10):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(64, 32)
        self.outc = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3)
        diffY = x2.size()[2] - x.size()[2]
        diffX = x2.size()[3] - x.size()[3]
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up1(x)
        x = self.up2(x)
        diffY = x1.size()[2] - x.size()[2]
        diffX = x1.size()[3] - x.size()[3]
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up2(x)
        logits = self.outc(x)
        return logits

def spectral_angle_mapper_loss(x, y, eps=1e-8):
    dot = torch.sum(x * y, dim=1)
    norm_x = torch.norm(x, p=2, dim=1)
    norm_y = torch.norm(y, p=2, dim=1)
    cos_theta = dot / (norm_x * norm_y + eps)
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    sam = torch.acos(cos_theta)
    return sam.mean()

# ==========================================
# 2. HSI Dataset
# ==========================================

class Hyspecnet11kDataset(Dataset):
    def __init__(self, root, transform=None, is_train=True):
        self.root = Path(root)
        self.transform = transform
        self.is_train = is_train
        
        # Depending on whether data is flat or in 'data' directory
        data_dir = self.root / 'data' if (self.root / 'data').exists() else self.root
        
        # Load all .npy files
        self.image_paths = sorted(data_dir.rglob('*.npy'))
        assert len(self.image_paths) > 0, f'Found {len(self.image_paths)} images in {root}.'
        
        # Just use a fixed split for train/test
        split_idx = int(len(self.image_paths) * 0.9)
        if self.is_train:
            self.image_paths = self.image_paths[:split_idx]
        else:
            self.image_paths = self.image_paths[split_idx:]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        impath = self.image_paths[index]
        img = np.load(impath).astype(np.float32) # (C, H, W)
        
        img = img / (img.max() + 1e-8)  # normalize
        img = torch.from_numpy(img)
        
        # Dummy label for task adapter demonstration since hyspecnet11k here only provides .npy
        # Real tasks would load from a label file mapping
        label = index % 9 
        return img, label

# ==========================================
# 3. Training Loop (Rate-Distortion + Task)
# ==========================================

def configure_optimizers(net, args):
    parameters = {
        n for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }
    params_dict = dict(net.named_parameters())
    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer

def train_one_epoch(
    model, task_model, criterion_task, train_dataloader, optimizer, task_optimizer, aux_optimizer, epoch, args, lr_scheduler
):
    model.train()
    task_model.train()
    device = next(model.parameters()).device
    
    alpha = args.alpha
    lambda_m = args.lambda_m

    for i, (d, target) in enumerate(train_dataloader):
        d = d.to(device)
        target = target.to(device)
        B, C, H, W = d.shape
        num_pixels = B * H * W
        
        optimizer.zero_grad()
        task_optimizer.zero_grad()
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
        
        pred_logits = task_model(x_hat)
        task_loss = criterion_task(pred_logits, target)
        
        # Loss: compression loss + task loss
        loss = args.lmbda * d_human + bpp_loss + lambda_m * task_loss
        
        loss.backward()

        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm) 
        
        optimizer.step()
        task_optimizer.step()
        
        aux_loss = model.module.aux_loss() if torch.cuda.device_count() > 1 else model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if (i+1) % 10 == 0:
            print(
                f"Train epoch {epoch}: [{i*len(d)}/{len(train_dataloader.dataset)}]\t"
                f"Loss: {loss.item():.4f} | Bpppb: {bpppb_loss.item():.4f} | "
                f"MSE: {mse_loss.item():.4f} | SAM: {sam_loss.item():.4f} | "
                f"Task Loss: {task_loss.item():.4f}"
            )

def test_epoch(model, task_model, criterion_task, test_dataloader, epoch, args):
    model.eval()
    task_model.eval()
    device = next(model.parameters()).device
    
    alpha = args.alpha
    lambda_m = args.lambda_m

    loss_sum = bpppb_sum = mse_sum = sam_sum = psnr_sum = task_sum = 0
    count = 0
    
    with torch.no_grad():
        for d, target in test_dataloader:
            d = d.to(device)
            target = target.to(device)
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
            
            pred_logits = task_model(x_hat)
            task_loss = criterion_task(pred_logits, target)
            
            loss = args.lmbda * d_human + bpp_loss + lambda_m * task_loss
            
            loss_sum += loss.item()
            bpppb_sum += bpppb_loss.item()
            mse_sum += mse_loss.item()
            sam_sum += sam_loss.item()
            psnr_sum += psnr
            task_sum += task_loss.item()
            count += 1
            
    print(
        f"Test epoch {epoch}: "
        f"Loss: {loss_sum/count:.4f} | Bpppb: {bpppb_sum/count:.4f} | "
        f"MSE: {mse_sum/count:.4f} | SAM: {sam_sum/count:.4f} | "
        f"PSNR: {psnr_sum/count:.4f} | Task Loss: {task_sum/count:.4f}"
    )
    return loss_sum/count

def main(argv):
    parser = argparse.ArgumentParser(description="HSI DCAE + Task Adapter Training")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Training dataset")
    parser.add_argument("--test_dataset", type=str, default="/media/nercms/EXTERNAL_USB/satellite_data/HSI_data/hyspecnet11k/test_data", help="Test dataset")
    parser.add_argument("-e", "--epochs", default=50, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument("--aux-learning-rate", default=1e-3, type=float)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-batch-size", type=int, default=2)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=100.0, help="Bit-rate distortion parameter")
    parser.add_argument("--lambda_m", type=float, default=10.0, help="Task loss weight")
    parser.add_argument("--alpha", type=float, default=0.1, help="SAM vs MSE weight")
    parser.add_argument("--clip_max_norm", default=1.0, type=float)
    parser.add_argument("--save_path", type=str, default="./hsi_runs/")
    parser.add_argument("--in_channels", type=int, default=202)
    parser.add_argument("--task_type", type=str, default="classification", choices=["classification", "segmentation"])
    parser.add_argument("--eval_freq", type=int, default=1, help="Evaluate every N epochs")
    args = parser.parse_args(argv)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dataset
    train_dataset = Hyspecnet11kDataset(args.dataset, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    test_dataset = Hyspecnet11kDataset(args.test_dataset, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)
    
    # Model
    net = DCAE(in_channels=args.in_channels).to(device)
    optimizer, aux_optimizer = configure_optimizers(net, args)
    
    # Task Model
    if args.task_type == "classification":
        task_model = Classifier(in_channels=args.in_channels, num_classes=9).to(device)
        criterion_task = nn.CrossEntropyLoss()
    else:
        task_model = UNet(in_channels=args.in_channels, num_classes=10).to(device)
        criterion_task = nn.CrossEntropyLoss(ignore_index=0)
        
    task_optimizer = optim.Adam(task_model.parameters(), lr=args.learning_rate)
    
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [40], gamma=0.1, last_epoch=-1)
    
    os.makedirs(args.save_path, exist_ok=True)
    
    print(f"Starting Training on {device} with in_channels={args.in_channels}")
    
    for epoch in range(args.epochs):
        train_one_epoch(net, task_model, criterion_task, train_dataloader, optimizer, task_optimizer, aux_optimizer, epoch, args, lr_scheduler)
        lr_scheduler.step()
        
        if (epoch + 1) % args.eval_freq == 0:
            test_epoch(net, task_model, criterion_task, test_dataloader, epoch, args)
        
        torch.save({
            "epoch": epoch,
            "state_dict": net.state_dict(),
            "task_state_dict": task_model.state_dict(),
        }, os.path.join(args.save_path, "hsi_checkpoint.pth.tar"))

if __name__ == "__main__":
    main(sys.argv[1:])
