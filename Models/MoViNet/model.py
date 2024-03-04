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

from official.vision.configs import video_classification
from official.projects.movinet.configs import movinet as movinet_configs
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_layers
from official.projects.movinet.modeling import movinet_model

tf.get_logger().setLevel('ERROR')
absl.logging.set_verbosity(absl.logging.ERROR)
mpl.rcParams.update({
    'font.size': 10,
})

# with tf.io.gfile.GFile('Models/labels.txt') as f:
#   lines = f.readlines()
#   KINETICS_600_LABELS_LIST = [line.strip() for line in lines]
#   KINETICS_600_LABELS = tf.constant(KINETICS_600_LABELS_LIST)

# def get_top_k(probs, k=5, label_map=KINETICS_600_LABELS):
#   """Outputs the top k model labels and probabilities on the given video."""
#   top_predictions = tf.argsort(probs, axis=-1, direction='DESCENDING')[:k]
#   top_labels = tf.gather(label_map, top_predictions, axis=-1)
#   top_labels = [label.decode('utf8') for label in top_labels.numpy()]
#   top_probs = tf.gather(probs, top_predictions, axis=-1).numpy()
#   return tuple(zip(top_labels, top_probs))

# def predict_top_k(model, video, k=5, label_map=KINETICS_600_LABELS):
#   """Outputs the top k model labels and probabilities on the given video."""
#   outputs = model.predict(video[tf.newaxis])[0]
#   probs = tf.nn.softmax(outputs)
#   return get_top_k(probs, k=k, label_map=label_map)

backbone = movinet.Movinet(model_id='a0')
model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
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
  model = movinet_model.MovinetClassifier(
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
  model = build_classifier(backbone, 100, freeze_backbone=False)

  num_epochs = 3
  train_steps = len(train_videos) // batch_size
  total_train_steps = train_steps * num_epochs
  test_steps = len(val_videos) // batch_size

  loss_obj = tf.keras.losses.CategoricalCrossentropy(
  from_logits=True,
  label_smoothing=0.1)

  metrics = [
  tf.keras.metrics.TopKCategoricalAccuracy(
      k=1, name='top_1', dtype=tf.float32),
  tf.keras.metrics.TopKCategoricalAccuracy(
      k=5, name='top_5', dtype=tf.float32),
  ]

  initial_learning_rate = 0.01
  learning_rate = tf.keras.optimizers.schedules.CosineDecay(
  initial_learning_rate, decay_steps=total_train_steps,
  )
  optimizer = tf.keras.optimizers.RMSprop(
  learning_rate, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0)

  model.compile(loss=loss_obj, optimizer=optimizer, metrics=metrics)


  checkpoint_path = "Models/MoViNet/data/training_1/cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)

  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                verbose=1)

  results = model.fit(train_dataset, validation_data=val_videos, epochs=num_epochs, 
                  steps_per_epoch=train_steps, validation_steps=test_steps, callbacks=[cp_callback])

  print(results)

  loss, accuracy = model.evaluate(test_videos, batch_size=batch_size)
  print(loss, accuracy)
# print(model.summary())


#print(video.shape)

# outputs = predict_top_k(model, video)

# for label, prob in outputs:
#   print(label, prob)