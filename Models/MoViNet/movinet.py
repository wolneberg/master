import numpy as np
import pandas as pd
import tensorflow as tf
import random
import cv2

import os
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

from lr_finder import LRFinder


num_frames = 64 
batch_size = 4
resolution = 172
num_classes = 100

# Preprocessing of WLASL
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
    f = open('WLASL/data/wlasl_class_list.txt', 'r')
    gloss_set = []
    for line in f:
        new_line = line.strip().split('\t')
        new_line[0] = int(new_line[0])
        gloss_set.append(new_line)
    f.close()
    gloss_set = pd.DataFrame(gloss_set, columns=['gloss_id', 'gloss'])
    glosses = gloss_set.merge(glosses, on='gloss')
    #glosses = glosses.drop('gloss', axis=1)
    return glosses

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

def format_dataset(video_list, gloss_set, missing, train=False):
    frame_list = list(filter(lambda x: len(x[1])>0,
                             list(map(lambda video_id: [f'{video_id:05}', create_frames_from_each_video_file(f'{video_id:05}', missing)],video_list))))
    frame_set = pd.DataFrame(frame_list, columns=['video_id', 'frames'])
    frame_set = frame_set.merge(gloss_set, on='video_id')
    target = tf.one_hot(frame_set['gloss_id'], num_classes) # 100 er num_classes
    frame_set = frame_set['frames'].to_list()
    formatted = tf.data.Dataset.from_tensor_slices((frame_set, target))
    if train:
        formatted = formatted.shuffle(buffer_size=formatted.cardinality())
    return formatted

print('Getting the subset of WLASL100, missing videos and glosses')
train_videos = get_video_subset(f'nslt_{num_classes}', 'train')
val_videos = get_video_subset(f'nslt_{num_classes}', 'val')
test_videos = get_video_subset(f'nslt_{num_classes}', 'test')
missing = get_missing_videos()
glosses = get_glosses()

print('Formatting the datasets')
train_dataset = format_dataset(train_videos, glosses, missing, train=True)
val_dataset = format_dataset(val_videos, glosses, missing)
test_dataset = format_dataset(test_videos, glosses, missing)

train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

print('Done formatting the dataset ', val_dataset.take(1))

#Building and fine-tuning the MoViNet model
backbone = movinet.Movinet(model_id='a0')
model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
model.build([1, 1, 1, 1, 3])

checkpoint_dir = 'Models/MoViNet/data/movinet_a0_base'
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(checkpoint_path)
status.assert_existing_objects_matched()

def build_classifier(backbone, num_classes, freeze_backbone=False):
    """Builds a classifier on top of a backbone model."""
    model = movinet_model.MovinetClassifier(backbone=backbone,num_classes=num_classes)
    model.build([batch_size, num_frames, resolution, resolution, 3])

    if freeze_backbone:
        for layer in model.layers[:-1]:
            layer.trainable = False
        model.layers[-1].trainable = True
    return model

model = build_classifier(backbone, num_classes)

print('Done building the classifier')

#Training and evaluation on WLASL100
def train_and_eval(trainset, train_videos, valset, test_videos):
    # Wrap the backbone with a new classifier to create a new classifier head
    # with num_classes outputs (101 classes for UCF101).
    # Freeze all layers except for the final classifier head.
    num_epochs = 20
    train_steps = len(train_videos) // batch_size
    total_train_steps = train_steps * num_epochs
    test_steps = len(valset) // batch_size

    #loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)

    metrics = [
    tf.keras.metrics.TopKCategoricalAccuracy(
        k=1, name='top_1', dtype=tf.float32),
    tf.keras.metrics.TopKCategoricalAccuracy(
        k=5, name='top_5', dtype=tf.float32),
    ]
    

    loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # optimizer = tf.keras.optimizers.Adam(learning_rate = 0.005)
    
    initial_learning_rate = 0.05
    learning_rate = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps=total_train_steps,)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0)
    
    model.compile(loss=loss_obj, optimizer=optimizer, metrics=metrics)
    #callbacks = [tf.keras.callbacks.TensorBoard(),]
     

    checkpoint_path = "Models/MoViNet/data/training_6/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                verbose=1)
    
    results = model.fit(trainset, validation_data=valset, epochs=num_epochs, callbacks=[cp_callback], validation_freq=1,verbose=2)

    print(results)
    
    eval_results = model.evaluate(test_videos, batch_size=batch_size)
    print(eval_results)
    model.save('Models/MoViNet/data/MoViNet_a0_WLASL100_6', save_format='tf')

def find_learning_rate(trainset):
# Wrap the backbone with a new classifier to create a new classifier head
    # with num_classes outputs (101 classes for UCF101).
    # Freeze all layers except for the final classifier head

    lr_finder = LRFinder()

    #loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)

    loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # optimizer = tf.keras.optimizers.Adam(learning_rate = 0.005)
    
    # initial_learning_rate = 0.05
    # learning_rate = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps=total_train_steps,)
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0)
    
    model.compile(loss=loss_obj, optimizer="adam")
    
    _ = model.fit(trainset, epochs=5, callbacks=[lr_finder],verbose=False)
    lr_finder.plot()


#Training and evaluation on WLASL100
print('Finding learning rate')
find_learning_rate(train_dataset)

# print('Starting training and evaluation')
# train_and_eval(train_dataset, train_videos, val_dataset, test_dataset)