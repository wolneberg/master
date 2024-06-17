import datetime
import torch
import torch.nn as nn
import torchvision

import WLASL.transforms
from WLASL.extraction import get_glosses, get_video_subset
from WLASL.pytorch_format import WLASLDataset, make_dataset
from Models.x3d.x3d import train


def main():
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)


    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = 100
    num_epochs = 50
    batch_size = 8
    resolution = 182
    frames = 20
    model_name = 'x3d_l'
    name = f'{model_name}-{datetime.datetime.now().strftime("%m%d%Y-%H%M%S")}'

    print("-----------------------------------------")
    print("-----------------------------------------")
    print(f"Device: {device}")
    print(f"Num classes: {num_classes}")
    print(f"Number of epochs: {num_epochs}")
    print(f'Frames: {frames}, resolution: {resolution}')
    print(f"Batch size: {batch_size}")
    print("-----------------------------------------")
    print("-----------------------------------------")

    # Check if GPU is availabe and required libraries are installed
    print(torch.cuda.is_available())
    

    # Load data
    print('Getting train videos...')
    train_videos = get_video_subset(f'nslt_{num_classes}', 'train')
    print('Getting validation videos...')
    val_videos = get_video_subset(f'nslt_{num_classes}', 'val')

    print('Getting glosses')
    glosses = get_glosses()

    train_dataset = make_dataset('WLASL/videos/', train_videos, glosses)
    val_dataset = make_dataset('WLASL/videos/', val_videos, glosses)

    train_dataset = WLASLDataset(train_dataset, transform=torchvision.transforms.Compose([
                WLASL.transforms.VideoFilePathToTensor(max_len=frames, fps=12, padding_mode='last'),
                WLASL.transforms.VideoResize([resolution, resolution])]))
    val_dataset = WLASLDataset(val_dataset, transform=torchvision.transforms.Compose([
        WLASL.transforms.VideoFilePathToTensor(max_len=frames, fps=12,padding_mode='last'), 
        WLASL.transforms.VideoResize([resolution,resolution])]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    del train_videos
    del val_videos
    model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
    for params in model.parameters():
            params.requires_grad = True

    model.fc = nn.Linear(in_features=2048, out_features=100, bias=True)  
    train(device, model, num_epochs, train_loader, val_loader, resolution, frames, name, model_name)


if __name__ == "__main__":
    main()