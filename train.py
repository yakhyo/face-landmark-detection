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


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)
    logging.info(f'Save checkpoint to {filename}')


def train_one_epoch(dataloader, pfld_backbone, auxiliarynet, criterion, optimizer, epoch):
    losses = AverageMeter("Loss")
    pfld_backbone.train()
    auxiliarynet.train()

    for batch_idx, (image, landmark_gt, attribute_gt, euler_angle_gt) in enumerate(dataloader):
        image = image.to(device)
        attribute_gt = attribute_gt.to(device)
        landmark_gt = landmark_gt.to(device)
        euler_angle_gt = euler_angle_gt.to(device)

        features, landmarks = pfld_backbone(image)
        angle = auxiliarynet(features)
        weighted_loss, loss = criterion(attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks)

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        losses.update(loss.item())

        if (batch_idx + 1) % 10 == 0:
            logging.info(
                f"Epoch [{epoch:>3}] Batch [{batch_idx:>3}/{len(dataloader):<3}] - "
                f"Loss: {loss.item():<7.4f} (Avg: {losses.avg:<7.4f})"
            )

    logging.info(f"Epoch {epoch} - Training loss: {losses.avg:.4f}")
    return losses.avg


def validate(val_loader, pfld_backbone, auxiliarynet):
    pfld_backbone.eval()
    auxiliarynet.eval()
    losses = []

    with torch.no_grad():
        for image, landmark_gt, attribute_gt, euler_angle_gt in val_loader:
            image = image.to(device)
            landmark_gt = landmark_gt.to(device)

            _, landmark = pfld_backbone(image)
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
        handlers=[logging.FileHandler('training.log', mode='w'), logging.StreamHandler()]
    )

    pfld_backbone = PFLDInference()
    pfld_backbone = pfld_backbone.to(device)

    auxiliarynet = AuxiliaryNet()
    auxiliarynet = auxiliarynet.to(device)

    criterion = PFLDLoss()
    optimizer = torch.optim.Adam(
        [
            {'params': pfld_backbone.parameters()},
            {'params': auxiliarynet.parameters()}
        ],
        lr=args.base_lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience)
    best_val_loss = float('inf')
    start_epoch = 0

    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, weights_only=False)
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

    train_dataset = WLFWDatasets(args.train_data, transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False
    )

    val_dataset = WLFWDatasets(args.val_data, transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    for epoch in range(start_epoch, args.num_epochs + 1):
        train_loss = train_one_epoch(
            train_loader, pfld_backbone, auxiliarynet, criterion, optimizer, epoch
        )
        val_loss = validate(val_loader, pfld_backbone, auxiliarynet)

        ckpt = {
            'epoch': epoch,
            'pfld_backbone': pfld_backbone.state_dict(),
            'auxiliarynet': auxiliarynet.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_loss': val_loss
        }

        save_checkpoint(ckpt, os.path.join(args.save_dir, 'last_ckpt.pth'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(ckpt, os.path.join(args.save_dir, 'best_ckpt.pth'))
            logging.info(f"Best checkpoint updated at epoch {epoch} with val_loss: {best_val_loss:.4f}")

        scheduler.step(val_loss)
        logging.info(f"Epoch {epoch}: Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")


def parse_args():
    """
    Parses command-line arguments for training the PFLD model.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="PFLD Training Configuration")

    # Training settings
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers (default: 4)")
    parser.add_argument("--base-lr", type=float, default=1e-4, help="Initial learning rate (default: 1e-4)")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-6,
        help="Weight decay for optimizer (default: 1e-6)"
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=40,
        help="Number of epochs with no improvement before reducing LR (default: 40)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=500,
        help="Total number of training epochs (default: 500)"
    )

    # Checkpoints & Resume
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoint/",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to resume training from a checkpoint"
    )

    # Dataset paths
    parser.add_argument(
        "--train-data",
        type=str,
        default="./data/train_data/list.txt",
        help="Path to the training dataset list file"
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default="./data/test_data/list.txt",
        help="Path to the validation dataset list file"
    )

    # Batch sizes
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training (default: 256)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
