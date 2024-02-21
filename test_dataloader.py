import os
import torch
from torch.utils.data.dataloader import default_collate
# local import
import transform
from dataloader import GestureDataset


def test_train_dataset():
    transform_train = transform.VideoClassificationPresetTrain(crop_size=(112, 112), resize_size=(128, 171))
    dataset = GestureDataset(
        os.path.join(os.getcwd(), "dataset"),
        transform=transform_train,
        sample_length=16,
        is_training=True,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=default_collate,
    )
    _ = next(iter(data_loader))


def test_test_dataset():
    transform_eval = transform.VideoClassificationPresetEval(crop_size=(112, 112), resize_size=(128, 171))
    dataset = GestureDataset(
        os.path.join(os.getcwd(), "dataset"),
        transform=transform_eval,
        sample_length=16,
        is_training=False,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=default_collate,
    )
    _ = next(iter(data_loader))

if __name__ == "__main__":
    test_train_dataset()
    test_test_dataset()