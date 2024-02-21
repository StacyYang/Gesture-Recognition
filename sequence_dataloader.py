import os
from dataloader import GestureDataset
from config import SequenceClasses


class GestureSequenceDataset(GestureDataset):
    def __init__(self, data_root, transform, sample_length, is_training):
        self.data_root = data_root
        self.transform = transform
        self.sample_length = sample_length
        self.is_training = is_training
        # List all video files in the data_root directory and its subfolders
        self.video_files = self._list_video_files(data_root)

    @staticmethod
    def _label_to_class(video_path: str) -> int:
        label = video_path.split(os.sep)[-2]
        # Get the class label from the parent folder of the video
        class_label = SequenceClasses[label]
        assert 0 <= class_label < len(SequenceClasses)
        return class_label
