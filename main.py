import hashlib
import datetime
import os
import time

import torch
import torch.utils.data
import torchvision
import torchvision.datasets.video_utils
from torch import nn
from torch.utils.data.dataloader import default_collate

# local imports
from dataloader import GestureDataset
import transform
import utils
from config import Classes


def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("clips/s", utils.SmoothedValue(window_size=10, fmt="{value:.3f}"))

    header = f"Epoch: [{epoch}]"
    for video, target, _ in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        video, target = video.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(video)
            loss = criterion(output, target)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = video.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["clips/s"].update(batch_size / (time.time() - start_time))
        lr_scheduler.step()


def evaluate(model, criterion, data_loader, device, print_freq):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    num_processed_samples = 0
    # Group and aggregate output of a video
    num_videos = len(data_loader.dataset)
    num_classes = len(Classes)
    agg_preds = torch.zeros((num_videos, num_classes), dtype=torch.float32, device=device)
    agg_targets = torch.zeros((num_videos), dtype=torch.int32, device=device)
    with torch.inference_mode():
        for video, target, video_idx in metric_logger.log_every(data_loader, print_freq, header):
            video = video.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(video)
            loss = criterion(output, target)

            # Use softmax to convert output into prediction probability
            preds = torch.softmax(output, dim=1)
            for b in range(video.size(0)):
                idx = video_idx[b].item()
                agg_preds[idx] += preds[b].detach()
                agg_targets[idx] = target[b].detach().item()

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = video.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size

    metric_logger.synchronize_between_processes()

    print(
        " * Clip Acc@1 {top1.global_avg:.3f} Clip Acc@5 {top5.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5
        )
    )
    # Reduce the agg_preds and agg_targets from all gpu and show result
    agg_preds = utils.reduce_across_processes(agg_preds)
    agg_targets = utils.reduce_across_processes(agg_targets, op=torch.distributed.ReduceOp.MAX)
    agg_acc1, agg_acc5 = utils.accuracy(agg_preds, agg_targets, topk=(1, 5))
    print(" * Video Acc@1 {acc1:.3f} Video Acc@5 {acc5:.3f}".format(acc1=agg_acc1, acc5=agg_acc5))
    return metric_logger.acc1.global_avg


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    # Data loading code
    print("Loading data")
    val_resize_size = tuple(args.val_resize_size)
    val_crop_size = tuple(args.val_crop_size)
    train_resize_size = tuple(args.train_resize_size)
    train_crop_size = tuple(args.train_crop_size)

    traindir = os.path.join(args.data_path, "train")
    valdir = os.path.join(args.data_path, "val")

    print("Loading training data")
    st = time.time()
    transform_train = transform.VideoClassificationPresetTrain(crop_size=train_crop_size, resize_size=train_resize_size)

    dataset = GestureDataset(
        args.data_path,
        transform=transform_train,
        sample_length=16,
        is_training=True,
    )

    print("Took", time.time() - st)

    print("Loading validation data")
    transform_test = transform.VideoClassificationPresetEval(crop_size=val_crop_size, resize_size=val_resize_size)

    dataset_test = GestureDataset(
        args.data_path,
        transform=transform_test,
        sample_length=16,
        is_training=False,
    )

    classes_cnt = [0 for _ in range(len(Classes))]
    for sample in dataset_test:
        classes_cnt[sample[1]] += 1

    print("Per-class sample count for evaluation:", classes_cnt)

    print("Creating data loaders")

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        val_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
        collate_fn=default_collate,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=val_sampler,
        pin_memory=True,
        collate_fn=default_collate,
    )

    print("Creating model")
    model = torchvision.models.get_model(args.model, weights=args.weights)
    num_classes = len(Classes)
    model.fc = nn.Linear(512, num_classes)
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    iters_per_epoch = len(data_loader)
    lr_milestones = [iters_per_epoch * (m - args.lr_warmup_epochs) for m in args.lr_milestones]
    main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma)

    if args.lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * args.lr_warmup_epochs
        args.lr_warmup_method = args.lr_warmup_method.lower()
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )

        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        evaluate(model, criterion, data_loader_test, device=device, print_freq=args.print_freq)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, args.print_freq, scaler)
        evaluate(model, criterion, data_loader_test, device=device, print_freq=args.print_freq)
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Video Classification Training", add_help=add_help)

    parser.add_argument("--model", default="r2plus1d_18", type=str, help="model name")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")

    parser.add_argument("--data-path", default=os.path.join(os.getcwd(), "dataset"), type=str, help="dataset path")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=24, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=45, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument("--lr-milestones", nargs="+", default=[20, 30, 40], type=int, help="decrease lr on milestones")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-warmup-epochs", default=10, type=int, help="the number of epochs to warmup (default: 10)")
    parser.add_argument("--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)")
    parser.add_argument("--lr-warmup-decay", default=0.001, type=float, help="the decay for lr")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="output", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    parser.add_argument(
        "--val-resize-size",
        default=(128, 171),
        nargs="+",
        type=int,
        help="the resize size used for validation (default: (128, 171))",
    )
    parser.add_argument(
        "--val-crop-size",
        default=(112, 112),
        nargs="+",
        type=int,
        help="the central crop size used for validation (default: (112, 112))",
    )
    parser.add_argument(
        "--train-resize-size",
        default=(128, 171),
        nargs="+",
        type=int,
        help="the resize size used for training (default: (128, 171))",
    )
    parser.add_argument(
        "--train-crop-size",
        default=(112, 112),
        nargs="+",
        type=int,
        help="the random crop size used for training (default: (112, 112))",
    )

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)