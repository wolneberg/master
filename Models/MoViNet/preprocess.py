print('start')
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import cv2

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import PIL
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tqdm
import absl.logging

print('før official')

import tensorflow_models as tfm
import official as of

def get_video_subset(wlasl_samples, subset):
    videos = pd.read_json(f'WLASL/data/{wlasl_samples}.json').transpose()
    train_videos = videos[videos['subset'].str.contains(subset)].index.values.tolist()
    return train_videos

def get_missing_videos():
    f = open('WLASL/data/missing.txt', 'r')
    missing = []
    for line in f:
      missing.append(line.strip())
    f.close()
    return missing

def get_glosses():
    glosses = pd.read_json('WLASL/data/WLASL_v0.3.json')
    glosses = glosses.explode('instances').reset_index(drop=True).pipe(
  lambda x: pd.concat([x, pd.json_normalize(x['instances'])], axis=1)).drop(
    'instances', axis=1)[['gloss', 'video_id']]
        
    return glosses

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
def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 15):
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


def create_frames_from_each_video_file(video_id, missing):
  if video_id not in missing:
    frames = frames_from_video_file(f'WLASL/videos/{video_id}.mp4', 64)
    return frames
  return []

def format_dataset(video_list, gloss_set, missing):
  # frame_list = []
  # for video_id in video_list:
  #   frames = create_frames_from_each_video_file(f'{video_id:05}')
  #   if len(frames)>0:
  #     frame_list.append([video_id, frames])
  frame_list = list(filter(lambda x: len(x[1])>0, 
                      list(map(lambda video_id: [f'{video_id:05}', create_frames_from_each_video_file(f'{video_id:05}', missing)], 
                               video_list))))
  frame_set = pd.DataFrame(frame_list, columns=['video_id', 'frames'])
  # print(frame_set.dtypes)
  # print(gloss_set.dtypes)
  frame_set = frame_set.merge(gloss_set, on='video_id')
  target = frame_set.pop('gloss').to_list()
  frame_set = frame_set['frames'].to_list()
  print('ready to format')
  # print(list(frame_set.items())[:4])
  # print(frame_set[frame_set.isnull().any(axis=1)])
  formatted = tf.data.Dataset.from_tensor_slices((frame_set, target))
  #print(formatted.take(5))
  print(list(formatted.as_numpy_iterator())[:4])
  return formatted

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
print("ferdig")



backbone = of.projects.movinet.modeling.Movinet(model_id='a0')
model = of.projects.movinet.modeling.movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
model.build([1, 1, 1, 1, 3])

checkpoint_dir = 'Models/MoViNet/data/movinet_a0_base'
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(checkpoint_path)
status.assert_existing_objects_matched()
num_frames = 64
batch_size = 4
resolution = 172

def build_classifier(backbone, num_classes, freeze_backbone=False):
  """Builds a classifier on top of a backbone model."""
  model = of.projects.movinet.modeling.movinet_model.MovinetClassifier(
      backbone=backbone,
      num_classes=num_classes)
  model.build([batch_size, num_frames, resolution, resolution, 3])

  if freeze_backbone:
    for layer in model.layers[:-1]:
      layer.trainable = False
    model.layers[-1].trainable = True

  return model

def train_and_eval(format_dataset, train_videos, val_videos, test_videos, glosses, missing):
  train_dataset = format_dataset(train_videos, glosses, missing)

# Wrap the backbone with a new classifier to create a new classifier head
# with num_classes outputs (101 classes for UCF101).
# Freeze all layers except for the final classifier head.
  model = build_classifier(backbone, 100, freeze_backbone=True)

  # num_epochs = 3
  # train_steps = len(train_videos) // batch_size
  # total_train_steps = train_steps * num_epochs
  # test_steps = len(val_videos) // batch_size

  # loss_obj = tf.keras.losses.CategoricalCrossentropy(
  # from_logits=True,
  # label_smoothing=0.1)

  # metrics = [
  # tf.keras.metrics.TopKCategoricalAccuracy(
  #     k=1, name='top_1', dtype=tf.float32),
  # tf.keras.metrics.TopKCategoricalAccuracy(
  #     k=5, name='top_5', dtype=tf.float32),
  # ]

  # initial_learning_rate = 0.01
  # learning_rate = tf.keras.optimizers.schedules.CosineDecay(
  # initial_learning_rate, decay_steps=total_train_steps,
  # )
  # optimizer = tf.keras.optimizers.RMSprop(
  # learning_rate, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0)

  # model.compile(loss=loss_obj, optimizer=optimizer, metrics=metrics)


  # checkpoint_path = "Models/MoViNet/data/training_1/cp.ckpt"
  # checkpoint_dir = os.path.dirname(checkpoint_path)

  # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
  #                                               save_weights_only=True,
  #                                               verbose=1)

  # results = model.fit(train_dataset, validation_data=val_videos, epochs=num_epochs, 
  #                 steps_per_epoch=train_steps, validation_steps=test_steps, callbacks=[cp_callback])

  # print(results)

  # loss, accuracy = model.evaluate(test_videos, batch_size=batch_size)
  # print(loss, accuracy)

print('Training and evaluation...\n')
train_and_eval(format_dataset, train_videos, val_videos, test_videos, glosses, missing)