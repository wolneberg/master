import argparse
import os
from official.projects.movinet.tools import export_saved_model
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
from WLASL.extraction import get_glosses, get_video_subset
from WLASL.preprocess import format_dataset, get_frame_set
from Models.MoViNet.movinet_v2 import train
from utils.plot import plot_history
import numpy as np
import random
import datetime
import tensorflow as tf
import keras_tuner
from keras import backend as backend
from utils.top_k_predictions import calculate_accuracy

def main(args):
  name= f'Hyperparameter_{datetime.datetime.now().strftime("%m%d%Y-%H%M%S")}'
  model_id = args.model
  batch_size = args.batch_size
  epochs = args.epochs
  max_trials = args.max_trials
  num_classes = 100
  frames = 20
  frame_step = 2
  resolution = args.resolution
  hyperparameter_tuner = args.hyperparameter_tuner

  print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
  print(f'Model: {model_id}')
  print(f'Hyperparameter tuner: {hyperparameter_tuner}')
  print(f'name: {name}')
  print(f'Frames: {frames}, frame step: {frame_step}, resolution: {resolution}')
  print(f'Batch size: {batch_size}')
  print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

  def build_model(hp):  
    backend.clear_session()

    use_positional_encoding = model_id in {'a3', 'a4', 'a5'}
    stochastic_depth_drop_rate = hp.Float('stochastic_depth_drop_rate', min_value=0.0, max_value=1.0, step=0.1)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    
    backbone = movinet.Movinet(
      model_id=model_id, 
      causal=True,
      conv_type='2plus1d',
      se_type='2plus3d', 
      activation='hard_swish',
      gating_activation='hard_sigmoid', 
      use_positional_encoding = use_positional_encoding,
      stochastic_depth_drop_rate = stochastic_depth_drop_rate,
      use_external_states=False,
    )
    print(len(backbone.layers))

    unfreezLayers = hp.Int('freez_layers', min_value=0, max_value=len(backbone.layers), step=1)
    backbone.trainable = True
    for layer in backbone.layers[:-unfreezLayers]:
        layer.trainable = False
    model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600, output_states=True)

    # Load pre-trained weights on kinetics dataset
    checkpoint_dir = f'Models/MoViNet/Backbone/movinet_{model_id}_stream'
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(checkpoint_path)
    status.assert_existing_objects_matched()

    model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=100, dropout_rate=dropout_rate)
    
    inputs = tf.ones([1, frames, resolution, resolution, 3])
    model.build(inputs)

    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    initial_learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    # learning_rate = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps=total_train_steps)
    optimizer  = tf.keras.optimizers.RMSprop(initial_learning_rate, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0)
    # optimizer = tf.keras.optimizers.Adam(learning_rate = initial_learning_rate)
    model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])
    return model

  print('Getting train videos...\n')
  train_videos = get_video_subset(f'nslt_{num_classes}', 'train')
  print('Getting validation videos...\n')
  val_videos = get_video_subset(f'nslt_{num_classes}', 'val')
  print('Getting test videos...\n')
  test_videos = get_video_subset(f'nslt_{num_classes}', 'test')
  # test_videos = ["00635","01386","01397","05639","05735","07093","09945","11305","11311","12320","21933","22954","24960","26984",
  #                "31750","32246","33286","34685","34743","34836","36927","42953","45432","49173","51077","55375","57273","57943",
  #                "57944","63769","64209","64297","65506","68132","68446","68636","69225","69370","70237","07092"]
  print('Getting missing videos...\n')
  # missing = get_missing_videos()
  print('Getting glosses...\n')
  glosses = get_glosses()

  train_set = get_frame_set(train_videos, glosses, frames, resolution, frame_step)
  val_set = get_frame_set(val_videos, glosses, frames, resolution, frame_step)
  test_set = get_frame_set(test_videos, glosses, frames, resolution, frame_step)
  # glosses = get_less_glosses(train_set)

  print('Formatting datasets...')
  train_dataset = format_dataset(train_set, glosses, over=False)
  val_dataset = format_dataset(val_set, glosses, over=False)
  test_dataset = format_dataset(test_set, glosses, over=False)

  with tf.device("CPU"):
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(1)

  del train_videos, train_set, val_videos, val_set, glosses

  print('Hyperparameter tuning...\n')
  # Distributed searching
  # strategy = tf.distribute.MirroredStrategy()
  # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
  # with strategy.scope():

  # Set up hyperparameter tuner
  if (hyperparameter_tuner == 'b'):
    tuner = keras_tuner.BayesianOptimization(
      build_model,
      objective='val_accuracy',
      max_trials=max_trials,
      directory=f'Models/MoViNet/hyperparameter/{name}',
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
      directory=f'Models/MoViNet/hyperparameter/{name}',
      # distribution_strategy=tf.distribute.MirroredStrategy(),
      project_name='Movinet_tuning',
      seed = 123
    )
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(train_dataset, epochs=epochs, validation_data=val_dataset, verbose=2, callbacks=[stop_early])
    
  best_hps = tuner.get_best_hyperparameters()
  for hps in best_hps:
    print(f"Hyperparameters: {hps.values}")

  print('Training with the best hyperparameters...\n')
  best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
  model = tuner.hypermodel.build(best_hps)
  result = train(model, train_ds=train_dataset, val_ds=val_dataset, epochs=epochs, name=name, model_id=model_id)
  plot_history(result, name, 'MoViNet', f'{model_id}_stream')

  print("Evaluate... \n")
  model.evaluate(test_dataset, verbose=2)
  top_predictions = {}
  for element, label in test_dataset:
    logits = model.predict(element, verbose=0)
    outputs = tf.nn.softmax(logits)
    top_predictions[label.ref()] = tf.argsort(outputs, axis=-1, direction='DESCENDING')
  top_1, top_5 = calculate_accuracy(top_predictions, k=5)
  print(f'Top 1 accuracy: {top_1} and Top 5 accuracy: {top_5}')

  print("Saving model... \n")
  saved_model_dir = f'Models/MoViNet/models/{model_id}'
  os.path.dirname(saved_model_dir)
  
  input_shape = [1, frames, resolution, resolution, 3]
  print(input_shape)

  export_saved_model.export_saved_model(
    model=model,
    input_shape=input_shape,
    export_path=f'{saved_model_dir}/{name}',
    causal=True,
    bundle_input_init_states_fn=False)
  
  print("Converitng to TensorFlow Lite... \n")
  converter = tf.lite.TFLiteConverter.from_saved_model(f'{saved_model_dir}/{name}')
  tflite_model = converter.convert()
  os.path.dirname(f'{saved_model_dir}/lite/')
  with open(f'{saved_model_dir}/lite/{name}.tflite', 'wb') as f:
    f.write(tflite_model)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run hyperparameter tuning")
  parser.add_argument("-t", "--hyperparameter_tuner", help="The hyperparameter tuner. Hyperband (h), Bayesian (b)", required=False, default='b', type=str)
  parser.add_argument("-m", "--model", help="Model", required=False, default="a0", type=str)
  parser.add_argument("-mt", "--max_trials", help="Max hyperparameter tuning trials (Bayesian)", required=False, default=50,  type=int)
  parser.add_argument("-e", "--epochs", help="Number of epochs", required=False, default=50, type=int )
  parser.add_argument("-r", "--resolution", help="Hight and widht of the videos", required=False, default=172, type=int)
  parser.add_argument("-b", "--batch_size", help="Batch sieze", required=False, default=16, type=int)
  args = parser.parse_args()


  seed = 123
  tf.random.set_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED']=str(seed)

  main(args)