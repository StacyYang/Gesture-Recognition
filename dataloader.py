import os
import cv2
import torch
import numpy as np
from config import Classes

class GestureDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, transform, sample_length, is_training, num_eval_samples=100):
        self.data_root = data_root
        self.transform = transform
        self.sample_length = sample_length
        self.is_training = is_training
        # List all video files in the data_root directory and its subfolders
        video_files = self._list_video_files(data_root)

        ## train/val split
        # lock the seed to make it reproducible
        torch.manual_seed(1)
        if is_training:
            indices = torch.randperm(len(video_files))[num_eval_samples:].tolist()
        else:
            # pick the first 50 samples for eval
            indices = torch.randperm(len(video_files))[:num_eval_samples].tolist()
        self.video_files = [video_files[idx] for idx in indices]

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        video_tensors = self.read_video_frames(video_path)
        video_tensors = self.subsample_frame(video_tensors, self.sample_length, self.is_training)
        video_tensors = self.transform(video_tensors)

        class_label = self._label_to_class(video_path)
        return video_tensors, class_label.value, idx

    def __len__(self):
        return len(self.video_files)

    @staticmethod
    def subsample_frame(video_tensors, output_length, is_training=False):
        input_length = video_tensors.shape[0]
        if is_training:
            selected_index = (torch.randperm(input_length) % input_length)[:output_length]
            selected_index, _ = torch.sort(selected_index)
            data_length = selected_index.shape[0]
            if data_length < output_length:
                pad_length = output_length - data_length
                selected_index = torch.cat([selected_index, selected_index[-pad_length:]])
        else:
            selected_index = (torch.arange(output_length) / output_length * input_length).long()
        return video_tensors[selected_index]

    @staticmethod
    def _list_video_files(root_dir):
        video_files = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".mp4"):
                    video_files.append(os.path.join(root, file))
        return video_files

    @staticmethod
    def _label_to_class(video_path: str) -> int:
        label = video_path.split(os.sep)[-3]
        # Get the class label from the parent folder of the video
        class_label = Classes[label]
        assert 0 <= class_label < len(Classes)
        return class_label
        
    @staticmethod
    def read_video_frames(video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f'Error while trying to read video at {video_path}. Please check the path again.')
            return None

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.as_tensor(np.asarray(frame))
            frame = frame.permute(2, 0, 1)
            frames.append(frame)

        cap.release()
        return torch.stack(frames)
