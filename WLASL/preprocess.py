import pandas as pd
import numpy as np
import tensorflow as tf
import random
import cv2
import os

num_frames = 16

# # Preprocessing of WLASL
# def get_video_subset(wlasl_samples, subset):
#     videos = pd.read_json(f'WLASL/data/{wlasl_samples}.json').transpose()
#     train_videos = videos[videos['subset'].str.contains(subset)].index.values.tolist()
#     return train_videos

# def get_missing_videos():
#     f = open('WLASL/data/missing.txt', 'r')
#     missing = []
#     for line in f:
#         missing.append(line.strip())
#     f.close()
#     return missing

# def get_glosses():
#     glosses = pd.read_json('WLASL/data/WLASL_v0.3.json')
#     glosses = glosses.explode('instances').reset_index(drop=True).pipe(
#         lambda x: pd.concat([x, pd.json_normalize(x['instances'])], axis=1)).drop(
#         'instances', axis=1)[['gloss', 'video_id']]
#     f = open('WLASL/data/wlasl_class_list.txt', 'r')
#     gloss_set = []
#     for line in f:
#         new_line = line.strip().split('\t')
#         new_line[0] = int(new_line[0])
#         gloss_set.append(new_line)
#     f.close()
#     gloss_set = pd.DataFrame(gloss_set, columns=['gloss_id', 'gloss'])
#     glosses = gloss_set.merge(glosses, on='gloss')
#     #glosses = glosses.drop('gloss', axis=1)
#     return glosses

#Format videos into frames and format dataset
"""
https://www.tensorflow.org/tutorials/load_data/video#create_frames_from_each_video_file
"""
def format_frames(frame, output_size):
    """
        Pad and resize an image from a video.

        Args:
          frame: Image that needs to resized and padded. 
          output_size: Pixel size of the output frame image.

        Return:
          Formatted frame with padding of specified output size.
    """
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

"""
https://www.tensorflow.org/tutorials/load_data/video#create_frames_from_each_video_file
"""
def frames_from_video_file(video_path, n_frames, output_size = (172,172), frame_step = 15):
    """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
    """
    # Read each video frame by frame
    result = []
    src = cv2.VideoCapture(str(video_path))  

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)


    need_length = 1 + (n_frames - 1) * frame_step


    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    # ret is a boolean indicating whether read was successful, frame is the image itself
    ret, frame = src.read()
    result.append(format_frames(frame, output_size))
    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]
    return result


def create_frames_from_each_video_file(video_id):
    if os.path.isfile(f'WLASL/videos/{video_id}.mp4'):
        frames = frames_from_video_file(f'WLASL/videos/{video_id}.mp4', num_frames)
        return frames
    return []

    # if video_id not in missing:
    #     frames = frames_from_video_file(f'WLASL/videos/{video_id}.mp4', 64)
    #     return frames
    # return []

def get_less_glosses(train_set):
    grouped = train_set.groupby(['gloss']).count()
    filtered = grouped[grouped['gloss_id'] > 14]
    gloss_list = filtered.index.values.tolist()
    return gloss_list

def get_frame_set(video_list, gloss_set):
    frame_list = list(filter(lambda x: len(x[1])>0, list(map(lambda video_id: [f'{video_id:05}', create_frames_from_each_video_file(f'{video_id:05}')],video_list))))
    frame_set = pd.DataFrame(frame_list, columns=['video_id', 'frames'])
    frame_set = frame_set.merge(gloss_set, on='video_id')
    return frame_set

def format_dataset(frame_set, gloss_list, train=False, over=False):
    # temp_frame_list = list(filter(lambda x: len(x[1])>0,
    #                          list(map(lambda video_id: [f'{video_id:05}', create_frames_from_each_video_file(f'{video_id:05}')],video_list))))
    
    # frame_list = []
    # for video_id in video_list:
    #     frames = create_frames_from_each_video_file(f'{video_id:05}',missing)
    #     if len(frames)>0:
    #         frame_list.append(frames)
    # frame_set = pd.DataFrame(temp_frame_list, columns=['video_id', 'frames'])
    # frame_set = frame_set.merge(gloss_set, on='video_id')
    if (over):
        for index, video in frame_set.iterrows():
            if video['gloss'] not in gloss_list:  
                frame_set = frame_set.drop(index)


    # target = tf.one_hot(frame_set['gloss_id'], num_classes) # 100 er num_classes
    target= frame_set['gloss_id'].tolist()
    frames = frame_set['frames'].tolist()
    formatted = tf.data.Dataset.from_tensor_slices((frames, target))
    # formatted = list(frame_set.itertuples(index=False, name=None))
    # print(formatted[0])
    # if train:
    #     formatted = formatted.shuffle(formatted.cardinality())

    AUTOTUNE = tf.data.AUTOTUNE

    formatted = formatted.cache().shuffle(formatted.cardinality()).prefetch(buffer_size = AUTOTUNE)
    return formatted