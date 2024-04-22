import datetime
import torch
import torch.nn as nn
import torchvision
import argparse

import WLASL.transforms
from WLASL.extraction import get_glosses, get_video_subset
from WLASL.pytorch_format import WLASLDataset, make_dataset
from Models.x3d.x3d import train
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
import imageio
import numpy as np

def main():
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)


    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = 100
    num_epochs = 50
    batch_size = 8
    model_name = 'x3d_xs'
    name = f'{model_name}-{datetime.datetime.now().strftime("%d%m%Y-%H%M%S")}'

    print("-----------------------------------------")
    print("-----------------------------------------")
    print(f"Device: {device}")
    print(f"Num classes: {num_classes}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print("-----------------------------------------")
    print("-----------------------------------------")

    # Check if GPU is availabe and required libraries are installed
    print(torch.cuda.is_available())
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    frames_per_second = 30
    model_transform_params  = {
        "x3d_xs": {
            "side_size": 182,
            "crop_size": 182,
            "num_frames": 4,
            "sampling_rate": 12,
        },
        "x3d_s": {
            "side_size": 182,
            "crop_size": 182,
            "num_frames": 13,
            "sampling_rate": 6,
        },
        "x3d_m": {
            "side_size": 256,
            "crop_size": 256,
            "num_frames": 16,
            "sampling_rate": 5,
        }
    }
    transform_params = model_transform_params[model_name]

    transform =  ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(transform_params["num_frames"]),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=transform_params["side_size"]),
                CenterCropVideo(
                    crop_size=(transform_params["crop_size"], transform_params["crop_size"])
                )
            ]
        ),
    )
    clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"])/frames_per_second

    transform = torchvision.transforms.Compose([
        UniformTemporalSubsample(transform_params["num_frames"]),
        Lambda(lambda x: x/255.0),
        NormalizeVideo(mean, std),
        ShortSideScale(size=transform_params["side_size"]),
        CenterCropVideo(
            crop_size=(transform_params["crop_size"], transform_params["crop_size"])
        )
    ])

    

    # Load data
    print('Getting train videos...')
    train_videos = get_video_subset(f'nslt_{num_classes}', 'train')
    print('Getting validation videos...')
    val_videos = get_video_subset(f'nslt_{num_classes}', 'val')

    print('Getting glosses')
    glosses = get_glosses()

    train_dataset = make_dataset('WLASL/videos/', train_videos, glosses)
    val_dataset = make_dataset('WLASL/videos/', val_videos, glosses)

    # train_dataset = WLASLDataset(train_dataset, transform=transform)
    # val_dataset = WLASLDataset(val_dataset, transform=transform)
    train_dataset = WLASLDataset(train_dataset, transform=torchvision.transforms.Compose([
                WLASL.transforms.VideoFilePathToTensor(max_len=50, fps=12, padding_mode='last'),
                        UniformTemporalSubsample(transform_params["num_frames"]),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=transform_params["side_size"]),
                CenterCropVideo(
                    crop_size=(transform_params["crop_size"], transform_params["crop_size"])
                )]))
    val_dataset = WLASLDataset(val_dataset, transform=torchvision.transforms.Compose([
                WLASL.transforms.VideoFilePathToTensor(max_len=50, fps=12, padding_mode='last'),
                        UniformTemporalSubsample(transform_params["num_frames"]),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=transform_params["side_size"]),
                CenterCropVideo(
                    crop_size=(transform_params["crop_size"], transform_params["crop_size"])
                )]))

    # def to_gif(images, label):
    #     converted_images = np.clip(images * 255, 0, 255).numpy().astype(np.uint8)
    #     imageio.mimsave(f'./{label}.gif', converted_images, format='GIF', fps=12)
    
    # to_gif(train_dataset[132][0][0], train_dataset[132][1])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    del train_videos
    del val_videos
    model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
    for params in model.parameters():
            params.requires_grad = True
    # layers = list(model.blocks.children())
    # model.classifier[1] = nn.Conv3d(1024, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1)) 
    # model.fc = layers[-1]
    model.fc = nn.Linear(in_features=2048, out_features=100, bias=True)  
    train(device, model, num_epochs, train_loader, val_loader, name, model_name)


if __name__ == "__main__":
    main()