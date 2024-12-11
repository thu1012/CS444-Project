import math
import sys
from typing import Iterable

import torch
from torch.utils.tensorboard import SummaryWriter




def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, log_writer=None,
                    args=None):
    model.train(True)

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 40

    optimizer.zero_grad()  # Clear gradients at the beginning of each epoch

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(data_loader):

        # lr 
        if data_iter_step % 5 == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():  
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()  # Clear gradients for this step

        loss.backward()  # Perform backpropagation
        optimizer.step()  # Update the parameters

        torch.cuda.synchronize()  # Sync for accurate timing

        if data_iter_step % print_freq == 0:
            print(f'{header} Step [{data_iter_step}/{len(data_loader)}] Loss: {loss_value:.4f}')

        lr = optimizer.param_groups[0]["lr"]

        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value, epoch_1000x)

    print(f'Epoch {epoch} finished, total steps: {len(data_loader)}')

