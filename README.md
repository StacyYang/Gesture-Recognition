# Gesture Recognition Project

Gesture recognition project using R3D, M3C and R(2+1) models. This repo is based on [video classification exampe](https://github.com/pytorch/vision/tree/main/references/video_classification) from torchvision.

## Instructions

Please prepare the dataset before launching training/evaluation jobs.

### Training/Debug using single GPU

Use `--worker 0` to debug dataloader using single thread.
```bash
python main.py --model r2plus1d_18 --weights R2Plus1D_18_Weights.KINETICS400_V1  --output-dir output --workers 0
```

### Training using multiple GPUs

Example command of training models on 4 GPUs:

```bash
torchrun --nproc_per_node=4 main.py --model r2plus1d_18 --weights R2Plus1D_18_Weights.KINETICS400_V1  --output-dir output
```

### Evaluation

Note: Please run `git lfs fetch && git lfs pull` if you're running the program at the first time after downloading, which fixes the error of `_pickle.UnpicklingError: invalid load key, 'v'.`.

Evaluate a pretrained `r3d_18` model on single gesture recognition dataset.
```bash
python main.py --model r2plus1d_18 --output-dir output --test-only --resume logs/output_r2plus1d_18/checkpoint.pth
```

### Evaluate on Gesture Sequence Dataset

Note: Please run `git lfs fetch && git lfs pull` if you're running the program at the first time after downloading.

Use distributed evaluation since the dataset is quite big.
```bash
torchrun --nproc_per_node=4  eval_sequence.py --model r2plus1d_18 --resume logs/output_r2plus1d_18/checkpoint.pth --workers 4
```

### Demo on Gesture Sequence Recognition

Note: Please run `git lfs fetch && git lfs pull` if you're running the program at the first time after downloading.

Specify the input video path via `--input-video` (use `--device cpu` if you are running on a cpu-only machine):
```bash
python demo.py --model r2plus1d_18 --resume logs/output_r2plus1d_18/checkpoint.pth --input-video gesture_sequences/abort_hello/abort_hello_1.mp4 --sample-frames 32

python demo.py --model r3d_18 --resume logs/output_r3d_18/checkpoint.pth --input-video demo_circle_hello_stop_warn.mp4 --sample-frames 64 --device cpu

python demo.py --model r2plus1d_18 --resume logs/output_r2plus1d_18/checkpoint.pth --input-video dataset/turn/videos/turn_5_5.mp4 --pred-single --sample-frames 16 --device cpu
```

Example output
```bash
Creating model
Loading video

Making Prediction:
abort
hello
```

## Prepare dataset

### Gesture Recognition
Add the video data into the `dataset/` folder, and each subfolder corresponds to a category. Each class contains `videos` folder storing all the videos in `mp4` format. 

```bash
dataset/
    Class_A/
        videos/
            class_a_video_1.map4
            class_a_video_2.map4
            ...
    Class_B/
        videos/
            class_b_video_1.map4
            ...
    ...
```

An example gesture recognition dataset can be downloaded [here](https://drive.google.com/file/d/1h6xfkx7rMn1cmrYKQl92A9pq0K2ug0-I/view?usp=sharing).

```bash
# install gdown for downloading from gcloud
pip install gdown
# download the dataset
gdown 1h6xfkx7rMn1cmrYKQl92A9pq0K2ug0-I
# unzip the dataset
unzip gestures_basic_extracted.zip & mv gestures_basic_extracted dataset
```

### Gesture Sequence Recognition

Add the video data into the `gesture_sequences/` folder, and each subfolder corresponds to a category, containing the videos in `mp4` format.
```bash
gesture_sequences/
    Class_A/
        class_a_video_1.map4
        class_a_video_2.map4
    Class_B/
    ...
```

An example gesture sequence recognition dataset can be downloaded [here](https://drive.google.com/file/d/1HqvW7ymvrLUPXHlSu2lQjnkW17bed7dH/view?usp=sharing).

```bash
# download the dataset
gdown 1HqvW7ymvrLUPXHlSu2lQjnkW17bed7dH
# unzip the dataset
unzip gesture_sequences.zip
```
