import argparse
import os
from official.projects.movinet.tools import export_saved_model
from WLASL.extraction import get_glosses, get_video_subset
from WLASL.preprocess import format_dataset, get_frame_set
from Models.MoViNet.movinet_v2 import compile, build_classifier, train, build_model_inference
from utils.plot import plot_history
import imageio
import numpy as np
import random
from tensorflow_docs.vis import embed
import datetime
import tensorflow as tf


import tensorflow as tf
from utils.top_k_predictions import calculate_accuracy

# print(tf.config.list_physical_devices())

def main(args):
  versions = {'a0': ['a0_base', 'a0_stream'], 'a1': ['a1_base', 'a1_stream'], 'a2': ['a2_base', 'a2_stream'], 'a3': ['a3_base', 'a3_stream'], 'a4': ['a4_base', 'a4_stream'], 'a5': ['a5_base', 'a5_stream']}

  name= f'{datetime.datetime.now().strftime("%m%d%Y-%H%M%S")}'
  model = 'movinet'
  model_id = versions.get(args.version)[args.stream]

  batch_size = args.batch_size
  num_classes = 100
  frames = 20
  frame_step = 2
  resolution = args.resolution
  epochs = args.epochs
  activation = 'softmax'
  stochatic_depth_drop_rate = args.stochatic_depth_drop_rate
  dropout_rate = args.dropout_rate
  unfreez_layers = args.unfreez_layers

  optimizer = 'rmsprop'
  learning_rate = args.learning_rate
  #For RMSprop
  rho=0.9 
  momentum=0.9
  epsilon=1.0
  clipnorm=1.0

  print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
  print(f'{name}, {model}, {model_id}')
  print(f'Classes: {num_classes}, version: {model_id}')
  print(f'epochs: {epochs}')
  print(f'activation function: {activation}')
  print(f'batch size: {batch_size}')
  print(f'Stochatic depth dropout rate: {stochatic_depth_drop_rate}')
  print(f'Dropout rate: {dropout_rate}')
  print(f'unFreez Layers: {unfreez_layers}')
  print(f'Frames: {frames}, frame step: {frame_step}, resolution: {resolution}')
  print(f'Optimizer: {optimizer}')
  print(f'Initial learning rate: {learning_rate}, Rho: {rho}, Momentum: {momentum}, Epsilon: {epsilon}, clipnorm: {clipnorm} ')
  print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

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

  print('Formatting train...')
  train_dataset = format_dataset(train_set, glosses, over=False)
  print('Formatting val...')
  val_dataset = format_dataset(val_set, glosses, over=False)
  print('Formatting test...')
  test_dataset = format_dataset(test_set, glosses, over=False)


  """Function to create gif from the frames extracted (verify that the input are correct)"""
  def to_gif(images, label):
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    imageio.mimsave(f'./{label}.gif', converted_images, fps=20)
    return embed.embed_file(f'gifs/{label}.gif')
  # video = 100 # the index of the video to create gif
  # to_gif(train_set.iloc[video]['frames'], train_set.iloc[video]['gloss'])

  print('Batching... \n')
  with tf.device("CPU"):
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(1)

  print('Training...\n')
  model = build_classifier(model_id, batch_size, frames, resolution, num_classes, unFreezLayers=unfreez_layers, dropout_rate=dropout_rate, stochastic_depth_drop_rate=stochatic_depth_drop_rate)
  compile(model, len(train_set), batch_size, epochs, optimizer)
  result = train(model, train_ds=train_dataset, val_ds=val_dataset, epochs=epochs, name=name, model_id=model_id)
  print(f"Valdation loss: {min(result.history['val_loss'])} validation accuracy {max(result.history['val_accuracy'])}")
  print('||||||||||||||||||||||||||||||||||||||||||||||')
  plot_history(result, name, 'MoViNet', model_id)

  print('Evaluate... \n')
  model.evaluate(test_dataset, verbose=2)
  
  top_predictions = {}
  for element, label in test_dataset:
    logits = model.predict(element, verbose=0)
    outputs = tf.nn.softmax(logits)
    top_predictions[label.ref()] = tf.argsort(outputs, axis=-1, direction='DESCENDING')
  top_1, top_5 = calculate_accuracy(top_predictions, k=5)
  print(f'Top 1 accuracy: {top_1} and Top 5 accuracy: {top_5}')

  if args.stream == 1:
    print('Make inference model... \n')
    model = build_model_inference(model_id, frames, resolution, name)

  print('Saving model... \n')
  saved_model_dir = f'Models/MoViNet/models/{model_id}'
  os.path.dirname(saved_model_dir)
  
  input_shape = [1, frames, resolution, resolution, 3]
  print(input_shape)
  input_image = tf.ones(input_shape)

  export_saved_model.export_saved_model(
    model=model,
    input_shape=input_shape,
    export_path=f'{saved_model_dir}/{name}',
    causal=args.stream == 1,
    bundle_input_init_states_fn=False)
  
  print('Convert to TensorFlow Lite... \n')
  converter = tf.lite.TFLiteConverter.from_saved_model(f'{saved_model_dir}/{name}')
  if args.stream==0: # This is needed for the base model as they contain 3D operations
      converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
      ]
  tflite_model = converter.convert()
  os.path.dirname(f'{saved_model_dir}/lite/')

  with open(f'{saved_model_dir}/lite/{name}.tflite', 'wb') as f:
    f.write(tflite_model)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run Movinet")
  parser.add_argument("-b", "--batch_size", help="batch size to use for training", default=4, required=False, type=int)
  parser.add_argument("-l", "--learning_rate", help="Learning rate", default=0.01, required=False, type=float)
  parser.add_argument("-d", "--dropout_rate", help="Dropout rate", required=False, default=0.0, type=float)
  parser.add_argument("-sd", "--stochatic_depth_drop_rate", help="tochatic_depth_drop_rate", required=False, default=0.0, type=float)
  parser.add_argument("-uf", "--unfreez_layers", help="un_freez_layers", required=False, default=20, type=int)
  parser.add_argument("-v", "--version", help="Version of movinet, a0, a1, a2", required=False, default='a0', type=str)
  parser.add_argument("-s", "--stream", help="Is the model a stream model", required=False, default=1, type=int)
  parser.add_argument("-e", "--epochs", help="Epochs, 1, 2, 20, 100", required=False, default=20, type=int)
  parser.add_argument("-r", "--resolution", help="Hight and width, eg 172, 224, 256 ..", required=False, default=172, type=int)
  
  seed = 123
  tf.random.set_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED']=str(seed)

  args = parser.parse_args()
  main(args)