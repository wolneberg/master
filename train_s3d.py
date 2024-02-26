import torch
import torchvision

import WLASL.transforms
from WLASL.extraction import get_glosses, get_video_subset
from WLASL.pytorch_format import WLASLDataset, make_dataset
from Models.S3D.s3d import train

num_classes = 100
num_epochs = 10
batch_size = 4
fine_tune = True
name = 'S3D_1'

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
            WLASL.transforms.VideoFilePathToTensor(max_len=50, fps=10, padding_mode='last'),
            WLASL.transforms.VideoResize([256, 256])]))
val_dataset = WLASLDataset(val_dataset, transform=torchvision.transforms.Compose([
    WLASL.transforms.VideoFilePathToTensor(max_len=50, fps=10,padding_mode='last'), 
    WLASL.transforms.VideoResize([256,256])]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle= True)

# Train
    
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training {name} on {device} with {num_classes} classes, {num_epochs} epochs, {batch_size} and fine_tune set to {fine_tune}')

train(num_epochs, num_classes, train_loader, val_loader, device, fine_tune, name)