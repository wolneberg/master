import torch
import torchvision
import argparse

import WLASL.transforms
from WLASL.extraction import get_glosses, get_video_subset
from WLASL.pytorch_format import WLASLDataset, make_dataset
from Models.S3D.s3d import train

# num_classes = 100
# num_epochs = 15
# batch_size = 4
fine_tune = False
# name = 'S3D_2'

def main(args):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    print("-----------------------------------------")
    print("-----------------------------------------")
    print(f"Device: {device}")
    print(f"Num classes: {args.num_classes}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Fine tune: {fine_tune}")
    print("-----------------------------------------")
    print("-----------------------------------------")

    # Check if GPU is availabe and required libraries are installed
    print(torch.cuda.is_available())

    # Load data
    print('Getting train videos...')
    train_videos = get_video_subset(f'nslt_{args.num_classes}', 'train')
    print('Getting validation videos...')
    val_videos = get_video_subset(f'nslt_{args.num_classes}', 'val')

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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle= True)

    # Train
    train(args.num_epochs, args.num_classes, train_loader, val_loader, device, fine_tune, args.name, trainable_layers=args.trainable_layers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S3D")
    parser.add_argument("-n", "--name", help="Name for this run", required=True)
    parser.add_argument("-e", "--num_epochs", help="number of epochs", default=10, required=False, type=int)
    parser.add_argument("-b", "--batch_size", help="batch size to use for training", default=4, required=False, type=int)
    parser.add_argument("-c", "--num_classes", help="number of classes from WLASL dataset", default=100, required=False, type=int)
    parser.add_argument("-t", "--trainable_layers", help="amount of layers that are unfrozen", required=False, default=0, type=int)
   
    args = parser.parse_args()
    main(args)