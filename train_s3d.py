from WLASL.extraction import get_glosses, get_video_subset
from WLASL.pytorch_format import WLASLDataset
from Models.S3D.s3d import train

num_classes = 100

# Load data
print('Getting train videos...')
train_videos = get_video_subset(f'nslt_{num_classes}', 'train')

print('Getting glosses')
glosses = get_glosses()

train_dataset = WLASLDataset('WLASL/videos/', train_videos, glosses)

print(train_dataset.__getitem__(2))
# Train
