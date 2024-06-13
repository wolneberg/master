import argparse
import os
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


# print(tf.config.list_physical_devices())

def main():
  # output_file_name = args.output_file
  name= f'{datetime.datetime.now().strftime("%m%d%Y-%H%M%S")}'
  modelname = 'i3d'

  batch_size = 4
  num_classes = 100
  frames = 20
  frame_step = 1
  resolution = 224
  epochs = 200
  activation = 'softmax'

  optimizer = 'RMSprop'
  learning_rate = 0.01
  unFreezLayers = 15

  #For RMSprop
  rho=0.9 
  momentum=0.9
  epsilon=1.0
  clipnorm=1.0

  # tf.debugging.set_log_device_placement(True)
  # tf.config.list_physical_devices('GPU')

  print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
  print(f'Hyperparameter')
  print(f'{name}, {modelname}')
  print(f'Classes: {num_classes}')
  print(f'epochs: {epochs}')
  print(f'activation function: {activation}')
  print(f'batch size: {batch_size}')
  print(f'Frames: {frames}, frame step: {frame_step}, resolution: {resolution}')
  print(f'Optimizer: {optimizer}')
  print(f'UnfreezLayers: {unFreezLayers}')
  print(f'Initial learning rate: {learning_rate}, Rho: {rho}, Momentum: {momentum}, Epsilon: {epsilon}, clipnorm: {clipnorm} ')
  print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

  # test_video = frames_from_video_file('WLASL/videos/00339.mp4', 10)
  # print(test_video.shape)

  print('Getting train videos...\n')
  train_videos = get_video_subset(f'nslt_{num_classes}', 'train')
  print('\nGetting validation videos...\n')
  val_videos = get_video_subset(f'nslt_{num_classes}', 'val')
  # print('\nGetting test videos...\n')
  # test_videos = get_video_subset(f'nslt_{num_classes}', 'test')
  # print('\nGetting missing videos...\n')
  # missing = get_missing_videos()
  print('Getting glosses...\n')
  glosses = get_glosses()

  train_set = get_frame_set(train_videos, glosses, frames, resolution, frame_step)
  val_set = get_frame_set(val_videos, glosses, frames, resolution, frame_step)
  # test_set = get_frame_set(test_videos, glosses, frames, resolution, frame_step)


  # glosses = get_less_glosses(train_set)

  print('formatting train...')
  train_dataset = format_dataset(train_set, glosses, over=False)
  print('formatting val...')
  val_dataset = format_dataset(val_set, glosses, over=False)
  # print('formatting test...')
  # test_dataset = format_dataset(test_set, glosses, over=False)


  def to_gif(images, label):
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    imageio.mimsave(f'./{label}_3.gif', converted_images, fps=20)
    return embed.embed_file(f'./{label}_3.gif')

  to_gif(train_set.iloc[200]['frames'], train_set.iloc[200]['gloss'])

  print('batching...')
  with tf.device("CPU"):
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    # test_dataset = test_dataset.batch(1)
  strategy = tf.distribute.MirroredStrategy()
  print('Number of devices: {}'.format(strategy.num_replicas_in_sync))



  print('Training...\n')

  with strategy.scope():
    model = build_classifier(batch_size=batch_size, num_frames=frames, resolution=resolution, num_classes=num_classes, unFreezLayers=unFreezLayers)
    compile(model, len(train_set), batch_size, epochs, optimizer)
  result = train(model, train_ds=train_dataset, val_ds=val_dataset, epochs=epochs, name=name)

  print('||||||||||||||||||||||||||||||||||||||||||||||')
  print(f"Valdation loss: {min(result.history['val_loss'])} validation accuracy {max(result.history['val_accuracy'])}")
  plot_history(result, name, 'i3d','1')
  # model.evaluate(test_dataset, verbose=2)
  saved_model_dir = f'Models/{modelname}/models'
  os.path.dirname(f'{saved_model_dir}')
  
  input_shape = [1, frames, resolution, resolution, 3]
  print(input_shape)
  input_image = tf.ones(input_shape)

  tf.saved_model.save(model, f'{saved_model_dir}/{name}')

  input_signature = [tf.TensorSpec([3, 3], tf.float32, name='x')]
  # Use from_function for tf functions
  onnx_model, _ = tf2onnx.convert.from_keras(model)
  onnx.save(onnx_model, f"{saved_model_dir}/{name}.onnx")
  
  # converter = tf.lite.TFLiteConverter.from_saved_model(f'{saved_model_dir}/{name}')
  # converter.target_spec.supported_ops = [
  #       tf.lite.OpsSet.TFLITE_BUILTINS,
  #       tf.lite.OpsSet.SELECT_TF_OPS
  #     ]
  # tflite_model = converter.convert()
  # os.path.dirname(f'{saved_model_dir}/lite')

  # with open(f'{saved_model_dir}/lite/{name}.tflite', 'wb') as f:
  #   f.write(tflite_model)


if __name__ == "__main__":


  tf.random.set_seed(123)
  random.seed(123)
  np.random.seed(123)
  os.environ['PYTHONHASHSEED']=str(123)

  main()