import os
import torch
from torch import nn
import torchvision

# local import
from dataloader import GestureDataset
from config import Classes
from model import SequenceModel, HeuristicFindTopNPostprocessing
import transform

def main(args):
    device = torch.device(args.device)

    print("Creating model")
    model = torchvision.models.get_model(args.model)
    num_classes = len(Classes)
    model.fc = nn.Linear(512, num_classes)
    model.to(device)
    model.eval()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        args.start_epoch = checkpoint["epoch"] + 1

    if not args.pred_single:
        model = SequenceModel(
            model,
            HeuristicFindTopNPostprocessing(
                output_len=None,
                pred_conversion=False,
                conf_thres = 0.3,
                vote_conf_by_count = True,
                select_by_conf = True,
            )
        )

    # load the inputs
    print("Loading video")
    video_frames = GestureDataset.read_video_frames(args.input_video)
    if args.sample_frames is not None:
        video_frames = GestureDataset.subsample_frame(video_frames, args.sample_frames)

    # transform the inputs
    val_resize_size = tuple(args.val_resize_size)
    val_crop_size = tuple(args.val_crop_size)
    transform_test = transform.VideoClassificationPresetEval(crop_size=val_crop_size, resize_size=val_resize_size)
    video_frames = transform_test(video_frames)


    # make predictions
    print("\nMaking Prediction:")
    video_frames = video_frames.to(device)
    preds = model(video_frames.unsqueeze(0))[0]
    if args.pred_single:
        pred = preds.argmax().int().item()
        print(Classes(pred).name)
    else:
        pred_seq = preds.int().tolist()
        for pred in pred_seq:
            print(Classes(pred).name)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Video Classification Training", add_help=add_help)

    parser.add_argument("--model", default="r2plus1d_18", type=str, help="model name")
    parser.add_argument("--input-video", type=str, help="input video path")
    parser.add_argument("--pred-single", action='store_true', default= False, help="input video path")
    parser.add_argument("--sample-frames", default=None, type=int, help="subsample frames")

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")

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
