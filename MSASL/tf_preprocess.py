import pandas as pd
import numpy as np
import tensorflow as tf
import random
import cv2
import os


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
    result = np.array(result, 'float32')[..., [2, 1, 0]]
    return result

def create_frames_from_each_video_file(video_name, num_frames, resolution, frame_step):
    if os.path.isfile(f'MSASL/videos/{video_name}.mp4'):
        frames = frames_from_video_file(f'MSASL/videos/{video_name}.mp4', num_frames, output_size=(resolution, resolution), frame_step=frame_step)
        return frames
    return []

def get_frame_set(num_frames, resolution, frame_step, csv_file):
    gloss_set = pd.read_csv(csv_file)
    video_list = gloss_set['video_id'].tolist()
    frame_list = list(filter(lambda x: len(x[1])>0, list(map(lambda video_id: [video_id, create_frames_from_each_video_file(video_id, num_frames=num_frames, resolution=resolution, frame_step=frame_step)],video_list))))
    frame_set = pd.DataFrame(frame_list, columns=['video_id', 'frames'])
    frame_set = frame_set.merge(gloss_set, on='video_id')
    return frame_set

def format_dataset(frame_set):
    target= frame_set['label'].tolist()
    frames = frame_set['frames'].tolist()
    formatted = tf.data.Dataset.from_tensor_slices((frames, target))

    AUTOTUNE = tf.data.AUTOTUNE

    formatted = formatted.cache().shuffle(formatted.cardinality()).prefetch(buffer_size = AUTOTUNE)
    return formatted