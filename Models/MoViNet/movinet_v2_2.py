import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras

import tensorflow as tf
import tensorflow_hub as hub

from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

model_id = 'a0'
num_frames = 20
batch_size = 32
resolution = 172
num_classes = 100
activation = 'softmax'

print('movinet kjører')
# print(f'model: {model_id}, classes: {num_classes}, batch: {batch_size}, resolution: {resolution}, activation: {activation}')



try:
  print(tf.config.list_physical_devices('GPU'))
  strategy = tf.distribute.MirroredStrategy()
  print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
except:
  print('gikk ikke å printe')



def build_classifier(batch_size, num_frames, resolution, backbone, num_classes, activation, freeze_backbone=False):
  """Builds a classifier on top of a backbone model."""
  model = movinet_model.MovinetClassifier(
      backbone=backbone,
      num_classes=num_classes, activation=activation)
  model.build([batch_size, num_frames, resolution, resolution, 3])
  if freeze_backbone:
    for layer in model.layers[:-1]:
        layer.trainable = False
    model.layers[-1].trainable = True
  return model

# with strategy.scope():

backbone = movinet.Movinet(model_id=model_id[:2])
# for layer in backbone.layers[:-3]:
#       layer.trainable = False

# Set num_classes=600 to load the pre-trained weights from the original model



# hub_url = "https://www.kaggle.com/models/google/movinet/frameworks/TensorFlow2/variations/a0-base-kinetics-600-classification/versions/3"

# encoder = hub.KerasLayer(hub_url, trainable=True)

# inputs = tf.keras.layers.Input(
#     shape=[None, None, None, 3],
#     dtype=tf.float32,
#     name='image')

# outputs = [batch_size, 100]
# outputs = encoder(dict(image=inputs))

# model = tf.keras.Model(inputs, outputs, name='movinet')
# model = build_classifier(batch_size=batch_size, activation=activation, backbone=model, num_frames=num_frames, resolution=resolution, num_classes=num_classes )

# Load pre-trained weights

# checkpoint_dir = f'Models/MoViNet/Backbone/movinet_{model_id}_base'
# checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
# checkpoint = tf.train.Checkpoint(model=model)
# status = checkpoint.restore(checkpoint_path)
# status.assert_existing_objects_matched()

x = model.layers[-1].output
print(x)
output = tf.keras.layers.Dense(units=num_classes, activation=activation)(x)
model = tf.keras.Model(inputs=model.inputs, outputs=output)

# model = build_classifier(batch_size, num_frames, resolution, backbone, num_classes, activation, freeze_backbone=False)

# loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  # optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
  # train_steps = len(train_videos) // batch_size
  # total_train_steps = train_steps * epochs
  # initial_learning_rate = 0.01
  # learning_rate = keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps=total_train_steps,)
  # optimizer = keras.optimizers.RMSprop(learning_rate, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0)

# model.compile(loss=loss_obj, optimizer='rmsprop', metrics=['accuracy'])

def train(model_id, batch_size, num_frames, resolution, num_classes, activation, train_ds, val_ds, epochs, optimizer, len_vid):
  print("Training a movinet model...")
  # backbone = movinet.Movinet(model_id=model_id[:2])
  # for layer in backbone.layers[:-3]:
  #       layer.trainable = False

  # # Set num_classes=600 to load the pre-trained weights from the original model
  # model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
  # model.build([None, None, None, None, 3])

  # Load pre-trained weights

  # checkpoint_dir = f'Models/MoViNet/Backbone/movinet_{model_id}'
  # checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
  # checkpoint = tf.train.Checkpoint(model=model)
  # status = checkpoint.restore(checkpoint_path)
  # status.assert_existing_objects_matched()

  # model = build_classifier(batch_size, num_frames, resolution, backbone, num_classes, activation, True)

  loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  # optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
  # train_steps = len_vid // batch_size
  # total_train_steps = train_steps * epochs
  # initial_learning_rate = 0.01
  # learning_rate = keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps=total_train_steps,)
  # optimizer = keras.optimizers.RMSprop(learning_rate, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0)
  

  model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])
  print(model.summary())
  return model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)






