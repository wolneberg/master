from WLASL.extraction import get_glosses, get_video_subset
from WLASL.preprocess import format_dataset, get_less_glosses, get_frame_set
# from Models.V3DCNN.v3dcnn import train
from Models.MoViNet.movinet_v2 import train

batch_size = 8
num_classes = 100
epochs = 20
name = 'movinet3'

# test_video = frames_from_video_file('WLASL/videos/00339.mp4', 10)
# print(test_video.shape)

print('Getting train videos...\n')
train_videos = get_video_subset(f'nslt_{num_classes}', 'train')
print('\nGetting validation videos...\n')
val_videos = get_video_subset(f'nslt_{num_classes}', 'val')
# # print('\nGetting test videos...\n')
# # test_videos = get_video_subset(f'nslt_{num_classes}', 'test')
# print('\nGetting missing videos...\n')
# missing = get_missing_videos()
print('Getting glosses...\n')
glosses = get_glosses()

train_set = get_frame_set(train_videos, glosses)
val_set = get_frame_set(val_videos, glosses)
gloss_list = get_less_glosses(train_set)

print('formatting train...')
train_dataset = format_dataset(train_set, gloss_list, over=True)
print('formatting val...')
val_dataset = format_dataset(val_set, gloss_list, over=True)
# # print(train_dataset.take(2))
# # print(train_dataset.element_spec)
# # test_dataset = format_dataset(test_videos, glosses, num_classes=num_classes, missing=missing)

print('batching...')
train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)

print('Training...\n')
train(train_dataset, val_dataset, epochs, name, train_videos)