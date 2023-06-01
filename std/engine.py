# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import dataclasses
import math
import random
import sys
import warnings
from contextlib import suppress
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from std.losses import DistillationLoss
import std.utils as utils
from std.models.std_mlp_mixer import STDMLPMixer


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, amp_autocast=None, n_mine_samples: int = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if n_mine_samples == 1:
        # We cannot pick joint vs marginal with sample size = 1.
        warnings.warn("Given `n_mine_samples` must be greater than 1 (if not 0), "
                      "changing the number to be set as default (batch-size)")
        n_mine_samples = None

    n_mine_samples = data_loader.batch_size if n_mine_samples is None else n_mine_samples
    mine_samples = None
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        mine_rnd = random.randint(0, samples.shape[0] - 1)
        mine_sample = samples[mine_rnd].expand(1, -1, -1, -1)

        if n_mine_samples > 1 and mine_samples.shape[0] < n_mine_samples:
            if mine_samples is None:
                mine_samples = mine_sample
            else:
                mine_samples = torch.cat((mine_samples, mine_sample), 0)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with amp_autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward(create_graph=is_second_order)
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if mine_samples:
        mine_samples = mine_samples.to(device, non_blocking=True)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats, mine_samples


@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with amp_autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@dataclasses.dataclass
class DatasetArgs:
    data_set: str = "CIFAR"
    data_path: str = "None"
    batch_size: int = 128
    num_workers: int = 10
    pin_mem = False
    input_size = 224
    drop: float = 0.0
    color_jitter = 0.4
    aa = "rand-m9-mstd0.5-inc1"
    train_interpolation = "bicubic"
    reprob: float = 0.25
    remode: str = "pixel"
    recount: int = 1


if __name__ == "__main__":
    from dataset import build_dataset

    device = torch.device("cuda")

    args = DatasetArgs()
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
    )

    data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
    )

    model = STDMLPMixer(image_size=args.input_size, channels=3, patch_size=16, dim=512, depth=1, dropout=args.drop, num_classes=100)
    model.to(device)

    amp_autocast = suppress  # do nothing
    max_accuracy = 0.0
    test_stats = evaluate(data_loader_val, model, device, amp_autocast=amp_autocast)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    max_accuracy = max(max_accuracy, test_stats["acc1"])
    print(f'Max accuracy: {max_accuracy:.2f}%')
