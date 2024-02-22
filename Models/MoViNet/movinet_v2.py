import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras

from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

model_id = 'a0'
num_frames = 16
batch_size = 8
resolution = 172
num_classes = 100
activation = 'softmax'

print('movinet kjører')

tf.keras.backend.clear_session()

# print(tf.config.list_physical_devices('GPU'))

# strategy = tf.distribute.MirroredStrategy()
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


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

backbone = movinet.Movinet(model_id=model_id)
# for layer in backbone.layers[:-3]:
#       layer.trainable = False

# Set num_classes=600 to load the pre-trained weights from the original model
model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
model.build([None, None, None, None, 3])

# Load pre-trained weights

checkpoint_dir = f'Models/MoViNet/data/movinet_{model_id}_base'
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(checkpoint_path)
status.assert_existing_objects_matched()

build_classifier(batch_size, num_frames, resolution, backbone, num_classes, activation, freeze_backbone=True)

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  # optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
  # train_steps = len(train_videos) // batch_size
  # total_train_steps = train_steps * epochs
  # initial_learning_rate = 0.01
  # learning_rate = keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps=total_train_steps,)
  # optimizer = keras.optimizers.RMSprop(learning_rate, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0)

model.compile(loss=loss_obj, optimizer='rmsprop', metrics=['accuracy'])

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

  plt.savefig(f'Models/MoViNet/data/results/{name}.png')

def train(train_ds, val_ds, epochs, name, train_videos):
    print("Training a movinet model...")
    # model = build_classifier(batch_size, num_frames, resolution, backbone, num_classes, activation)

    # loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # # optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    # # train_steps = len(train_videos) // batch_size
    # # total_train_steps = train_steps * epochs
    # # initial_learning_rate = 0.01
    # # learning_rate = keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps=total_train_steps,)
    # # optimizer = keras.optimizers.RMSprop(learning_rate, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0)

    # model.compile(loss=loss_obj, optimizer='rmsprop', metrics=['accuracy'])

    results = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)

    plot_history(results, name)





