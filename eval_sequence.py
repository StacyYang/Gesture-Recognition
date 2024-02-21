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
import utils
from config import Classes
from sequence_dataloader import GestureSequenceDataset
import transform
from model import SequenceModel, HeuristicFindTopNPostprocessing

def evaluate(model, data_loader, device, print_freq):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    num_processed_samples = 0
    with torch.inference_mode():
        for video, target, _ in metric_logger.log_every(data_loader, print_freq, header):
            video = video.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(video)

            acc1 = utils.pred_accuracy(output, target)
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = video.shape[0]
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            num_processed_samples += batch_size

    metric_logger.synchronize_between_processes()
    print(
        " * Clip Acc@1 {top1.global_avg:.3f}".format(
            top1=metric_logger.acc1
        )
    )
    return metric_logger.acc1.global_avg


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")
    val_resize_size = tuple(args.val_resize_size)
    val_crop_size = tuple(args.val_crop_size)

    print("Loading validation data")

    transform_test = transform.VideoClassificationPresetEval(crop_size=val_crop_size, resize_size=val_resize_size)
    dataset_test = GestureSequenceDataset(
        args.data_path,
        transform=transform_test,
        sample_length=args.seq_length,
        is_training=False,
    )
    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        val_sampler = torch.utils.data.SequentialSampler(dataset_test)

    print("Creating data loaders")
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=default_collate,
    )

    print("Creating model")
    model = torchvision.models.get_model(args.model, weights=args.weights)
    num_classes = len(Classes)
    model.fc = nn.Linear(512, num_classes)
    model.to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        args.start_epoch = checkpoint["epoch"] + 1

    model = SequenceModel(
        model,
        HeuristicFindTopNPostprocessing(
            output_len=2,
            pred_conversion=True,
            conf_thres=args.conf_thres,
            vote_conf_by_count=args.vote_conf_by_count,
            select_by_conf=args.select_by_conf,
        )
    )
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # We disable the cudnn benchmarking because it can noticeably affect the accuracy
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    evaluate(model, data_loader_test, device=device, print_freq=args.print_freq)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Video Classification Training", add_help=add_help)

    parser.add_argument("--model", default="r2plus1d_18", type=str, help="model name")
    parser.add_argument("--seq-length", default=32, type=int, help="video frames per sequence")
    parser.add_argument("--conf-thres", default=None, type=float, help="confidence threshold")
    parser.add_argument("--select-by-conf", default=False, action='store_true', help="select top prediction by confidence")
    parser.add_argument("--vote-conf-by-count", default=False, action='store_true', help="vote confidence score by count")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--data-path", default=os.path.join(os.getcwd(), "gesture_sequences"), type=str, help="dataset path")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=24, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")

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
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
