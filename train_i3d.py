import argparse
import os
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from Models.i3d.i3d import build_classifier, train, compile
from WLASL.extraction import get_glosses, get_video_subset
from WLASL.preprocess import format_dataset, get_frame_set
from utils.plot import plot_history
import imageio
import numpy as np
import random
from tensorflow_docs.vis import embed
import datetime
import tensorflow as tf
import tf2onnx
import onnx
from utils.top_k_predictions import calculate_accuracy

def main(args):
  # output_file_name = args.output_file
  name= f'{datetime.datetime.now().strftime("%m%d%Y-%H%M%S")}'
  modelname = 'i3d'
  batch_size = args.batch_size
  epochs = args.epochs
  num_classes = 100
  frames = 20
  frame_step = 2
  resolution = args.resolution
  num_classes = 100
  unfreez_layers = args.unfreez_layers
  dropout_rate = args.dropout_rate
  learning_rate = args.learning_rate

  print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
  print(f'Hyperparameter')
  print(f'{name}, {modelname}')
  print(f'Classes: {num_classes}')
  print(f'epochs: {epochs}')
  print(f'batch size: {batch_size}')
  print(f'Frames: {frames}, frame step: {frame_step}, resolution: {resolution}')
  print(f'UnfreezLayers: {unfreez_layers}')
  print(f'Dropouot rate: {dropout_rate}')
  print(f'Initial learning rate: {learning_rate}')
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
  model = build_classifier(num_frames=frames, resolution=resolution, num_classes=num_classes, unFreezLayers=unfreez_layers, dropout_rate=dropout_rate)
  compile(model, learning_rate=learning_rate)
  result = train(model, train_ds=train_dataset, val_ds=val_dataset, epochs=epochs, name=name)

  print('||||||||||||||||||||||||||||||||||||||||||||||')
  print(f"Valdation loss: {min(result.history['val_loss'])} validation accuracy {max(result.history['val_accuracy'])}")
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
  saved_model_dir = f'Models/{modelname}/models'
  os.path.dirname(f'{saved_model_dir}')
  
  input_shape = [1, frames, resolution, resolution, 3]
  print(input_shape)

  tf.saved_model.save(model, f'{saved_model_dir}/{name}')

  onnx_model, _ = tf2onnx.convert.from_keras(model)
  onnx.save(onnx_model, f"{saved_model_dir}/{name}.onnx")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run hyperparameter tuning")
  parser.add_argument("-e", "--epochs", help="Number of epochs", required=False, default=50, type=int )
  parser.add_argument("-r", "--resolution", help="Hight and widht of the videos", required=False, default=224, type=int)
  parser.add_argument("-b", "--batch_size", help="Batch sieze", required=False, default=16, type=int)
  parser.add_argument("-d", "--dropout_rate", help="Dropout rate for dense layer", required=False, default=0.0, type=float)
  parser.add_argument("-l", "--learning_rate", help="Learning rate", required=False, default=0.01, type=float)
  parser.add_argument('-ul', '--unfreez-layers', help="Number of layers to unfreez", required=False, default=0, type=int)
  args = parser.parse_args()

  seed = 123
  tf.random.set_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED']=str(seed)

  main(args)