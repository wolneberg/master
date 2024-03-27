import argparse
import os
from official.projects.movinet.tools import export_saved_model
from WLASL.extraction import get_glosses, get_video_subset
from WLASL.preprocess import format_dataset, get_less_glosses, get_frame_set
# from Models.V3DCNN.v3dcnn import train
from Models.MoViNet.movinet_v2 import compile, build_classifier, train, build_model_inference
from utils.plot import plot_history
import imageio
import numpy as np
from tensorflow_docs.vis import embed
# import tensorflow as tf
# import tf2onnx
# import onnx
import datetime


import tensorflow as tf, tf_keras
# import keras.backend as K

# print(tf.config.list_physical_devices())

def main(args):
  versions = {'a0': ['a0_base', 'a0_stream'], 'a2': ['a2_base', 'a2_stream']}

  # output_file_name = args.output_file
  name= datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
  model = 'movinet'
  model_id = versions.get('a0')[1]

  batch_size = 1
  num_classes = 100
  frames = 20
  frame_step = 2
  resolution = 172
  epochs = 20
  activation = 'softmax'
  num_unfreeze_layers = 10

  optimizer = 'rmsprop'
  learning_rate = 0.01
  #For RMSprop
  rho=0.9 
  momentum=0.9
  epsilon=1.0
  clipnorm=1.0

  # tf.debugging.set_log_device_placement(True)
  # tf.config.list_physical_devices('GPU')

  print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
  print(f'{name}, {model}, {model_id}')
  print(f'Classes: {num_classes}, version: {model_id}')
  print(f'epochs: {epochs}')
  print(f'activation function: {activation}')
  print(f' Number of unfreezed layers: {num_unfreeze_layers}')
  print(f'batch size: {batch_size}')
  print(f'Frames: {frames}, frame step: {frame_step}, resolution: {resolution}')
  print(f'Initial learning rate: {learning_rate}, Rho: {rho}, Momentum: {momentum}, Epsilon: {epsilon}, clipnorm: {clipnorm} ')
  print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

  print(f'{name}, epochs: {epochs}, classes: {num_classes}, batch size: {batch_size}')

  # test_video = frames_from_video_file('WLASL/videos/00339.mp4', 10)
  # print(test_video.shape)

  print('Getting train videos...\n')
  train_videos = get_video_subset(f'nslt_{num_classes}', 'train')
  print('\nGetting validation videos...\n')
  val_videos = get_video_subset(f'nslt_{num_classes}', 'val')
  print('\nGetting test videos...\n')
  # test_videos = get_video_subset(f'nslt_{num_classes}', 'test')
  # print('\nGetting missing videos...\n')
  # missing = get_missing_videos()
  print('Getting glosses...\n')
  glosses = get_glosses()

  train_set = get_frame_set(train_videos, glosses, frames, resolution, frame_step)
  val_set = get_frame_set(val_videos, glosses, frames, resolution, frame_step)
  # glosses = get_less_glosses(train_set)

  print('formatting train...')
  train_dataset = format_dataset(train_set, glosses, over=False)
  print('formatting val...')
  val_dataset = format_dataset(val_set, glosses, over=False)

  def to_gif(images, label):
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    imageio.mimsave(f'./{label}_2.gif', converted_images, fps=20)
    return embed.embed_file(f'./{label}_2.gif')
  # print(train_dataset.element_spec)
  # test_dataset = format_dataset(test_videos, glosses, num_classes=num_classes, missing=missing)
  # print(train_set.iloc[0]['frames'])
  # to_gif(train_set.iloc[100]['frames'], train_set.iloc[100]['gloss'])
  print('batching...')
  # with tf.device("CPU"):
  train_dataset = train_dataset.batch(batch_size)
  val_dataset = val_dataset.batch(batch_size)

  print('Training...\n')
  model = build_classifier(model_id, batch_size, frames, resolution, num_classes, num_unfreeze_layers, activation)
  compile(model, len(train_set), batch_size, epochs, optimizer)

  result = train(model, train_ds=train_dataset, val_ds=val_dataset, epochs=epochs, name=name, model_id=model_id)
  plot_history(result, name, model_id)

  saved_model_dir = f'Models/MoViNet/models/{model_id}/{name}'


  input_shape = (batch_size, frames, resolution, resolution, 3)
  input_shape_concrete = [1 if s is None else s for s in input_shape]


  model.build(input_shape_concrete)
  _ = model(tf.ones(input_shape_concrete))
  tf_keras.models.save_model(model, saved_model_dir)
  tf.saved_model.save(model,f'{saved_model_dir}_1')

  converter = tf.lite.TFLiteConverter.from_saved_model(f'{saved_model_dir}_1')
  tflite_model = converter.convert()
  
  with open(f'{saved_model_dir}_lite/{name}.tflite', 'wb') as f:
    f.write(tflite_model)
  model = build_model_inference(model_id, batch_size=batch_size, num_frames=frames, resolution=resolution,name=name, num_classes=num_classes, activation=activation )



  # Convert to saved model





  # input_shape_concrete = [1 if s is None else s for s in None]
  # _ = model(tf.ones(input_shape_concrete))
  tf_keras.models.save_model(model, f'{saved_model_dir}_2')
  # tf.saved_model.save(model, saved_model_dir)

  converter = tf.lite.TFLiteConverter.from_saved_model(f'{saved_model_dir}_2') # path to the SavedModel directory
  tflite_model = converter.convert()

  # Save the model.
  with open(f'{name}.tflite', 'wb') as f:
    f.write(tflite_model)


  export_saved_model.export_saved_model(
    model=model,
    input_shape=input_shape,
    export_path=saved_model_dir,
    causal=True,
    bundle_input_init_states_fn=False)
  
  converter = tf.lite.TFLiteConverter.from_saved_model(f'{saved_model_dir}_1')
  tflite_model = converter.convert()
  
  with open(f'{saved_model_dir}_lite/{name}.tflite', 'wb') as f:
    f.write(tflite_model)
  # try:
  #   converter = tf.lite.TFLiteConverter.from_keras_model(model)
  #   tflite_model = converter.convert()
  #   # tflite_model.save(f'tflite_{name}.keras')
  #   with open('Models/MoViNet/lite/model.tflite', 'wb') as f:
  #     f.write(tflite_model)
  # except:
  #   print('Was not able to save lite version from keras')

  # try:
  #   # Convert the model
  #   converter = tf.lite.TFLiteConverter.from_saved_model(f'Models/MoViNet/models/{name}') # path to the SavedModel directory
  #   tflite_model = converter.convert()

  #   # Save the model.
  #   with open('model.tflite', 'wb') as f:
  #     f.write(tflite_model)
  # except:
  #   print('Was not able to save lite version from saved models')
  # onnx_model, _ = tf2onnx.convert.from_keras(model, [batch_size, frames, resolution, resolution, 3], opset=13)
  # onnx.save(onnx_model, "dst/path/model.onnx")
  # new_model = tf.keras.models.load_model(f'Models/MoViNet/model/{name}.keras')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S3D")
    parser.add_argument("-n", "--name", help="Name for this run", required=True)
   
    args = parser.parse_args()
    main(args)