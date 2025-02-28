import argparse
import logging
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataset import WLFWDatasets
from models.pfld import PFLDInference, AuxiliaryNet
from utils.loss import PFLDLoss
from utils.general import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info(f'Save checkpoint to {filename}')

def train_one_epoch(train_loader, pfld_backbone, auxiliarynet, criterion, optimizer, epoch, batch_size):
    losses = AverageMeter("Loss")
    pfld_backbone.train()
    auxiliarynet.train()

    for batch_idx, (img, landmark_gt, attribute_gt, euler_angle_gt) in enumerate(train_loader, 1):
        img = img.to(device)
        attribute_gt = attribute_gt.to(device)
        landmark_gt = landmark_gt.to(device)
        euler_angle_gt = euler_angle_gt.to(device)

        features, landmarks = pfld_backbone(img)
        angle = auxiliarynet(features)
        weighted_loss, loss = criterion(attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks, batch_size)

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        losses.update(loss.item())

        if batch_idx % 10 == 0:
            logging.info(f"Epoch [{epoch}] Batch [{batch_idx}/{len(train_loader)}] - "
                         f"Loss: {loss.item():.4f} (Avg: {losses.avg:.4f})")

    logging.info(f"Epoch {epoch} - Training loss: {losses.avg:.4f}")
    return losses.avg

def validate(val_loader, pfld_backbone, auxiliarynet):
    pfld_backbone.eval()
    auxiliarynet.eval()
    losses = []

    with torch.no_grad():
        for img, landmark_gt, attribute_gt, euler_angle_gt in val_loader:
            img = img.to(device)
            landmark_gt = landmark_gt.to(device)

            _, landmark = pfld_backbone(img)
            loss = torch.mean(torch.sum((landmark_gt - landmark) ** 2, axis=1))
            losses.append(loss.item())

    avg_loss = np.mean(losses)
    logging.info(f"Validation: Average loss: {avg_loss:.4f}")
    return avg_loss

def main(args):
    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO,
        handlers=[logging.FileHandler('app.log', mode='w'), logging.StreamHandler()]
    )

    pfld_backbone = PFLDInference()
    pfld_backbone = pfld_backbone.to(device)

    auxiliarynet = AuxiliaryNet()
    auxiliarynet = auxiliarynet.to(device)

    criterion = PFLDLoss()
    optimizer = torch.optim.Adam([
        {'params': pfld_backbone.parameters()},
        {'params': auxiliarynet.parameters()}
    ], lr=args.base_lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience)
    best_val_loss = float('inf')
    start_epoch = args.start_epoch

    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
        pfld_backbone.load_state_dict(checkpoint["pfld_backbone"])
        auxiliarynet.load_state_dict(checkpoint["auxiliarynet"])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint.get("epoch", start_epoch) + 1
        best_val_loss = checkpoint.get("val_loss", best_val_loss)
        logging.info(f"Resumed training from epoch {start_epoch}")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = WLFWDatasets(args.dataroot, transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False
    )

    val_dataset = WLFWDatasets(args.val_dataroot, transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.workers
    )

    for epoch in range(start_epoch, args.end_epoch + 1):
        train_loss = train_one_epoch(
            train_loader, pfld_backbone, auxiliarynet, criterion, optimizer, epoch, args.train_batchsize
        )
        val_loss = validate(val_loader, pfld_backbone, auxiliarynet)

        checkpoint_state = {
            'epoch': epoch,
            'pfld_backbone': pfld_backbone.state_dict(),
            'auxiliarynet': auxiliarynet.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_loss': val_loss
        }

        save_checkpoint(checkpoint_state, os.path.join(str(args.snapshot), 'last_ckpt.pth'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(checkpoint_state, os.path.join(str(args.snapshot), 'best_ckpt.pth'))
            logging.info(f"Best checkpoint updated at epoch {epoch} with val_loss: {best_val_loss:.4f}")

        scheduler.step(val_loss)
        logging.info(f"Epoch {epoch}: Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description='pfld')
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--base_lr', default=1e-4, type=float)
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float)
    parser.add_argument("--lr_patience", default=40, type=int)
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=500, type=int)
    parser.add_argument('--snapshot', default='./checkpoint/', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--dataroot', default='./data/train_data/list.txt', type=str)
    parser.add_argument('--val_dataroot', default='./data/test_data/list.txt', type=str)
    parser.add_argument('--train_batchsize', default=256, type=int)
    parser.add_argument('--val_batchsize', default=256, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
