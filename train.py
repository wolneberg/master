from WLASL.preprocess import format_dataset, get_glosses, get_missing_videos, get_video_subset
from Models.MoViNet.movinet_v2 import train

batch_size = 4
num_classes = 100
epochs = 15

print('Getting train videos...\n')
train_videos = get_video_subset(f'nslt_{num_classes}', 'train')
print(len(train_videos))
print('\nGetting validation videos...\n')
val_videos = get_video_subset(f'nslt_{num_classes}', 'val')
print(len(val_videos))
print('\nGetting missing videos...\n')
missing = get_missing_videos()
print('Getting glosses...\n')
glosses = get_glosses()

train_dataset = format_dataset(train_videos, glosses, missing, num_classes, train=True)
val_dataset = format_dataset(val_videos, glosses, num_classes, missing)

train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)

print('Training...\n')
train(train_dataset, val_dataset, epochs)