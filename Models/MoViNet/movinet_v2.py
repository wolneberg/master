import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as plt

import os
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

model_id = 'a0'
num_frames = 64 
batch_size = 4
resolution = 172
num_classes = 100
activation = 'softmax'

tf.keras.backend.clear_session()

backbone = movinet.Movinet(model_id=model_id)
for layer in backbone.layers[:-3]:
      layer.trainable = False
print(backbone.summary())
# Set num_classes=600 to load the pre-trained weights from the original model
model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
model.build([None, None, None, None, 3])

# Load pre-trained weights

checkpoint_dir = f'Models/MoViNet/data/movinet_{model_id}_base'
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(checkpoint_path)
status.assert_existing_objects_matched()

def build_classifier(batch_size, num_frames, resolution, backbone, num_classes, activation):
  """Builds a classifier on top of a backbone model."""
  model = movinet_model.MovinetClassifier(
      backbone=backbone,
      num_classes=num_classes, activation=activation)
  model.build([batch_size, num_frames, resolution, resolution, 3])
  return model

def plot_history(history, name):
  """
    Plotting training and validation learning curves.

    Args:
      history: model history with all the metric measures
  """
  fig, (ax1, ax2) = plt.subplots(2)

  fig.set_size_inches(18.5, 10.5)

  # Plot loss
  ax1.set_title('Loss')
  ax1.plot(history.history['loss'], label = 'train')
  ax1.plot(history.history['val_loss'], label = 'test')
  ax1.set_ylabel('Loss')

  # Determine upper bound of y-axis
  max_loss = max(history.history['loss'] + history.history['val_loss'])

  ax1.set_ylim([0, np.ceil(max_loss)])
  ax1.set_xlabel('Epoch')
  ax1.legend(['Train', 'Validation']) 

  # Plot accuracy
  ax2.set_title('Accuracy')
  ax2.plot(history.history['accuracy'],  label = 'train')
  ax2.plot(history.history['val_accuracy'], label = 'test')
  ax2.set_ylabel('Accuracy')
  ax2.set_ylim([0, 1])
  ax2.set_xlabel('Epoch')
  ax2.legend(['Train', 'Validation'])

  plt.savefig(f'Models/MoViNet/data/results/{name}')

def train(train_ds, val_ds, epochs):
    model = build_classifier(batch_size, num_frames, resolution, backbone, num_classes, activation)

    print(model.summary())

    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

    model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])

    results = model.fit(train_ds, validation_data=val_ds, epochs=epochs, validation_freq=1, verbose=2)

    plot_history(results)





