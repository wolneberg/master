
import argparse
import os
from Models.i3d.I3D_model.i3d_inception import Inception_Inflated3d, conv3d_bn
from Models.i3d.i3d import train
from WLASL.extraction import get_glosses, get_video_subset
from WLASL.preprocess import format_dataset, get_frame_set
from utils.plot import plot_history
import numpy as np
import random
import datetime
import tensorflow as tf
from keras import backend
import keras_tuner
import tf2onnx
import onnx
from utils.top_k_predictions import calculate_accuracy


def main(args):
  name= f'Hyperparameter_{datetime.datetime.now().strftime("%m%d%Y-%H%M%S")}'
  model = 'i3d'
  batch_size = args.batch_size
  epochs = args.epochs
  max_trials = args.max_trials
  num_classes = 100
  frames = 20
  frame_step = 2
  resolution = args.resolution
  hyperparameter_tuner = args.hyperparameter_tuner
  num_classes = 100

  print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
  print(f'Hyperparameter')
  print(f'Hyperparameter optimizer: {hyperparameter_tuner}')
  print(f'{name}, {model}')
  print(f'Classes: {num_classes}')
  print(f'epochs: {epochs}, max trials: {max_trials}')
  print(f'batch size: {batch_size}')
  print(f'Frames: {frames}, frame step: {frame_step}, resolution: {resolution}')
  print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

  def build_model(hp):
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    unfreezLayers = hp.Int('freez_layers', min_value=0, max_value=len(model.layers), step=1)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')

    # I3D model implemented by Oana Ignat
    # Uses weight from pre-trained on imagenet and kinetics datasetes
    model = Inception_Inflated3d(
                  include_top=False,
                  endpoint_logit=False, 
                  weights='rgb_imagenet_and_kinetics',
                  input_shape=(20, 224, 224, 3),
                  classes=100)
    x = model.layers[-1].output
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = conv3d_bn(x, 100, 1, 1, 1, padding='same', 
            use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')
    num_frames_remaining = int(x.shape[1])
    x = tf.keras.layers.Reshape((num_frames_remaining, 100))(x)
    x = tf.keras.layers.Lambda(lambda x: backend.mean(x, axis=1, keepdims=False),
                output_shape=lambda s: (s[0], s[2]))(x)
    x = tf.keras.layers.Activation('softmax', name='prediction')(x)
    model = tf.keras.Model(inputs=model.inputs, outputs=x)

    model.trainable = True
    for layer in model.layers[:-unfreezLayers]:
        layer.trainable = False

    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])
    
    return model

  print('Getting train videos...\n')
  train_videos = get_video_subset(f'nslt_{num_classes}', 'train')
  print('Getting validation videos...\n')
  val_videos = get_video_subset(f'nslt_{num_classes}', 'val')
  print('Getting test videos...\n')
  test_videos = get_video_subset(f'nslt_{num_classes}', 'test')
  # print('\nGetting missing videos...\n')
  # missing = get_missing_videos()
  print('Getting glosses...\n')
  glosses = get_glosses()

  train_set = get_frame_set(train_videos, glosses, frames, resolution, frame_step)
  val_set = get_frame_set(val_videos, glosses, frames, resolution, frame_step)
  test_set = get_frame_set(test_videos, glosses, frames, resolution, frame_step)


  print('Formatting train... \n')
  train_dataset = format_dataset(train_set, glosses, over=False)
  print('Formatting val... \n')
  val_dataset = format_dataset(val_set, glosses, over=False)
  print('Formatting test... \n')
  test_dataset = format_dataset(test_set, glosses, over=False)

  print('Batching... \n')
  with tf.device("CPU"):
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(1)

  print('Hyperparameter tuning...\n')
  # Distributed search
  # strategy = tf.distribute.MirroredStrategy()
  # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

 # Set up hyperparameter tuner
  if (hyperparameter_tuner == 'b'):
    tuner = keras_tuner.BayesianOptimization(
      build_model,
      objective='val_accuracy',
      max_trials=max_trials,
      directory=f'Models/i3d/hyperparameter/{name}',
      # distribution_strategy=tf.distribute.MirroredStrategy(),
      project_name='Movinet_tuning',
      seed = 123
    )
    tuner.search(train_dataset, epochs=epochs, validation_data=val_dataset, verbose=2)
  elif(hyperparameter_tuner == 'h'):
    tuner = keras_tuner.Hyperband(
      build_model,
      objective='val_accuracy',
      max_epochs=epochs,
      directory=f'Models/i3d/hyperparameter/{name}',
      # distribution_strategy=tf.distribute.MirroredStrategy(),
      project_name='Movinet_tuning',
      seed = 123
    )
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(train_dataset, epochs=epochs, validation_data=val_dataset, verbose=2, callbacks=[stop_early])
    
  best_hps = tuner.get_best_hyperparameters()
  for hps in best_hps:
    print(f"Hyperparameters: {hps.values}")

  best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
  print(f"Best Hyperparameters: {best_hps.values}")

  model = tuner.hypermodel.build(best_hps)
  print('Training with the best hyperparameters...\n')
  result = train(model, train_ds=train_dataset, val_ds=val_dataset, epochs=epochs, name=name)
  plot_history(result, name, 'i3d','2')

  print("Evaluating... \n")
  top_predictions = {}
  for element, label in test_dataset:
    logits = model.predict(element, verbose=0)
    outputs = tf.nn.softmax(logits)
    top_predictions[label.ref()] = tf.argsort(outputs, axis=-1, direction='DESCENDING')
  top_1, top_5 = calculate_accuracy(top_predictions, k=5)
  print(f'Top 1 accuracy: {top_1} and Top 5 accuracy: {top_5}')
  model.evaluate(test_dataset, verbose=2)

  print("Saving model... \n")
  saved_model_dir = f'Models/i3d/models'
  os.path.dirname(f'{saved_model_dir}')
  
  tf.saved_model.save(model, f'{saved_model_dir}/{name}')
  onnx_model, _ = tf2onnx.convert.from_keras(model)
  onnx.save(onnx_model, f"{saved_model_dir}/{name}.onnx")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run hyperparameter tuning")
  parser.add_argument("-t", "--hyperparameter_tuner", help="The hyperparameter tuner. Hyperband (h), Bayesian (b)", required=False, default='b', type=str)
  parser.add_argument("-mt", "--max_trials", help="Max hyperparameter tuning trials (Bayesian)", required=False, default=50,  type=int)
  parser.add_argument("-e", "--epochs", help="Number of epochs", required=False, default=50, type=int )
  parser.add_argument("-r", "--resolution", help="Hight and widht of the videos", required=False, default=224, type=int)
  parser.add_argument("-b", "--batch_size", help="Batch sieze", required=False, default=16, type=int)
  args = parser.parse_args()


  seed = 123
  tf.random.set_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED']=str(seed)

  main(args)