from WLASL.preprocess import format_dataset, get_glosses, get_missing_videos, get_video_subset
from Models.MoViNet.model import train_and_eval

train_videos = get_video_subset('nslt_100', 'train')
val_videos = get_video_subset('nslt_100', 'val')
test_videos = get_video_subset('nslt_100', 'test')
missing = get_missing_videos()
glosses = get_glosses()

train_and_eval(format_dataset, train_videos, val_videos, test_videos, glosses, missing)