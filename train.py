from Models.MoViNet.preprocess import format_dataset, get_glosses, get_missing_videos, get_video_subset
from Models.MoViNet.model import train_and_eval

print('Getting train videos...\n')
train_videos = get_video_subset('nslt_100', 'train')
print('Getting validation videos...\n')
val_videos = get_video_subset('nslt_100', 'val')
print('Getting test videos...\n')
test_videos = get_video_subset('nslt_100', 'test')
print('Getting missing videos...\n')
missing = get_missing_videos()
print('Getting glosses...\n')
glosses = get_glosses()

print('Training and evaluation...\n')
train_and_eval(format_dataset, train_videos, val_videos, test_videos, glosses, missing)