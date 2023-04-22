from monai.metrics import DiceMetric, PSNRMetric

from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    Resized,
    Zoomd
)

from monai.config import print_config
from monai.metrics import DiceMetric, PSNRMetric
# from monai.networks.nets import SwinUNETR

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)


import torch
from monai.data import create_test_image_2d, list_data_collate, decollate_batch, DataLoader
from torch.utils.tensorboard import SummaryWriter
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.visualize import plot_2d_or_3d_image

train_transforms = Compose(
        [
            LoadImaged(keys=["A", "B", "infarct_label", "myo_label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["A", "B"],
                a_min=0,
                a_max=255,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Resized(
                keys=["A", "B"],
                spatial_size=[64, 64],
                mode='bilinear'
            ),
            Resized(
                keys=["infarct_label", "myo_label"],
                spatial_size=[64, 64],
                mode='nearest'
            ),
            RandFlipd(
                keys=["A", "B", "infarct_label", "myo_label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["A", "B", "infarct_label", "myo_label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["A", "B", "infarct_label", "myo_label"],
                prob=0.10,
                max_k=3,
            ),
        ]
    )
val_transforms = Compose(
    [
        LoadImaged(keys=["A", "B", "infarct_label", "myo_label"], ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=["A", "B"],
            a_min=0,
            a_max=255,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Resized(
            keys=["A", "B"],
            spatial_size=[64, 64],
            mode='bilinear'
        ),
        Resized(
            keys=["infarct_label", "myo_label"],
            spatial_size=[64, 64],
            mode='nearest'
        )
    ]
)


def create_data_loader(data_dir, batch_size, num_workers=8):
    # full
    split_json = "/dataset.json"
    datasets = data_dir + split_json

    datalist = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")
    train_ds = CacheDataset(
        data=datalist,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=num_workers,
    )
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=batch_size, shuffle=True)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=num_workers)
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)
    return train_loader, val_loader


if __name__ == "__main__":
    data_dir = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI_LGE"
    batch_size = 1
    train_loader, val_loader = create_data_loader(data_dir, batch_size)
    print("train loader length: ", len(train_loader))

