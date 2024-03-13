import torch
import torchvision
import argparse
import logging

import WLASL.transforms
from WLASL.extraction import get_glosses, get_video_subset
from WLASL.pytorch_format import WLASLDataset, make_dataset
from Models.S3D.s3d import train


# num_classes = 100
# num_epochs = 15
# batch_size = 4
fine_tune = False
max_len = 100
fps = 20
padding_mode = 'last'
num_frames = 20
frame_step = 2
# name = 'S3D_2'

def main(args):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    LOG_FILE = f"Models/S3D/s3d_outputs/{args.name}.txt"

    logging.basicConfig(level=logging.INFO, 
            format='[%(levelname)s] %(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(LOG_FILE),
                logging.StreamHandler()
            ])
    logger = logging.getLogger()

    logger.info("-----------------------------------------")
    logger.info("-----------------------------------------")
    logger.info(f"Device: {device}")
    logger.info(f"Num classes: {args.num_classes}")
    logger.info(f"Number of epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Fine tune: {fine_tune}")
    logger.info(f'Max len: {max_len}')
    logger.info(f'fps: {fps}')
    logger.info(f'Padding mode: {padding_mode}')
    logger.info("-----------------------------------------")
    logger.info("-----------------------------------------")

    # Check if GPU is availabe and required libraries are installed
    logger.info(torch.cuda.is_available())


    # Load data
    logger.info('Getting train videos...')
    train_videos = get_video_subset(f'nslt_{args.num_classes}', 'train')
    logger.info('Getting validation videos...')
    val_videos = get_video_subset(f'nslt_{args.num_classes}', 'val')

    logger.info('Getting glosses')
    glosses = get_glosses()

    train_dataset = make_dataset('WLASL/videos/', train_videos, glosses)
    val_dataset = make_dataset('WLASL/videos/', val_videos, glosses)

    train_dataset = WLASLDataset(train_dataset, transform=torchvision.transforms.Compose([
                WLASL.transforms.VideoFilePathToTensor(max_len=max_len, fps=fps, padding_mode=padding_mode),
                WLASL.transforms.VideoResize([256, 256])]))
    val_dataset = WLASLDataset(val_dataset, transform=torchvision.transforms.Compose([
        WLASL.transforms.VideoFilePathToTensor(max_len=max_len, fps=fps, padding_mode=padding_mode), 
        WLASL.transforms.VideoResize([256,256])]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle= True)

    # Train
    train(args.num_epochs, args.num_classes, train_loader, val_loader, device, fine_tune, args.name, trainable_layers=args.trainable_layers, logger=logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S3D")
    parser.add_argument("-n", "--name", help="Name for this run", required=True)
    parser.add_argument("-e", "--num_epochs", help="number of epochs", default=10, required=False, type=int)
    parser.add_argument("-b", "--batch_size", help="batch size to use for training", default=4, required=False, type=int)
    parser.add_argument("-c", "--num_classes", help="number of classes from WLASL dataset", default=100, required=False, type=int)
    parser.add_argument("-t", "--trainable_layers", help="amount of layers that are unfrozen", required=False, default=0, type=int)
   
    args = parser.parse_args()
    main(args)