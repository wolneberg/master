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
# from tensorflow_docs.vis import embed
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
  model_id = versions.get(args.version)[args.stream]

  batch_size = args.batch_size
  num_classes = 100
  frames = 20
  frame_step = 2
  resolution = 172
  epochs = args.epochs
  activation = 'softmax'
  stochatic_depth_drop_rate = args.dropout_rate

  optimizer = 'rmsprop'
  learning_rate = args.learning_rate
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
  print(f'batch size: {batch_size}')
  print(f'Stochatic depth dropout rate: {stochatic_depth_drop_rate}')
  print(f'Frames: {frames}, frame step: {frame_step}, resolution: {resolution}')
  print(f'Optimizer: {optimizer}')
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
  test_videos = get_video_subset(f'nslt_{num_classes}', 'test')
  # print('\nGetting missing videos...\n')
  # missing = get_missing_videos()
  print('Getting glosses...\n')
  glosses = get_glosses()

  train_set = get_frame_set(train_videos, glosses, frames, resolution, frame_step)
  val_set = get_frame_set(val_videos, glosses, frames, resolution, frame_step)
  test_set = get_frame_set(test_videos, glosses, frames, resolution, frame_step)



  # glosses = get_less_glosses(train_set)

  print('formatting train...')
  train_dataset = format_dataset(train_set, glosses, over=False)
  print('formatting val...')
  val_dataset = format_dataset(val_set, glosses, over=False)
  test_dataset = format_dataset(test_set, glosses, over=False)


  # def to_gif(images, label):
  #   converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
  #   imageio.mimsave(f'./{label}_3.gif', converted_images, fps=20)
  #   return embed.embed_file(f'./{label}_3.gif')
  # # test_dataset = format_dataset(test_videos, glosses, num_classes=num_classes, missing=missing)
  # # print(train_set.iloc[0]['frames'])
  # to_gif(train_set.iloc[100]['frames'], train_set.iloc[100]['gloss'])
  print('batching...')
  with tf.device("CPU"):
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(1)


  print('Training...\n')
  model = build_classifier(model_id, batch_size, frames, resolution, num_classes, activation, stochastic_depth_drop_rate=stochatic_depth_drop_rate)
  compile(model, len(train_set), batch_size, epochs, optimizer)

  result = train(model, train_ds=train_dataset, val_ds=val_dataset, epochs=epochs, name=name, model_id=model_id)
  print('||||||||||||||||||||||||||||||||||||||||||||||')
  print(f"Valdation loss: {min(result.history['val_loss'])} validation accuracy {max(result.history['val_accuracy'])}")
  plot_history(result, name, model_id)
  try:
    model.evaluate(test_dataset)
  except:
    print('could not evaluate')

  if model_id[2:] == '_stream':
    print('make inference model')
    model = build_model_inference(model_id, frames, resolution, name)
  print(model.summary())
  try:
    i=0
    for element, label in test_dataset:
      test_input = element
      print(element.shape)
      print(label)
      i +=1
      if i == 2:
        break

    print(element.shape)
    output = model(test_input, training=False)
    prediction = tf.argmax(output, -1)
    print(output)
    print(prediction)
  except:
    print('did not work')

  try: 
    init_states_fn = model.init_states
    init_states = init_states_fn(tf.shape(tf.ones(shape=[1, 1, 172, 172, 3])))

    all_logits = []

    # To run on a video, pass in one frame at a time
    states = init_states
    for frames, label in test_dataset.take(1):
      for clip in frames[0]:
        # Input shape: [1, 1, 172, 172, 3]
        clip = tf.expand_dims(tf.expand_dims(clip, axis=0), axis=0)
        logits, states = model.predict({**states, 'image': clip}, verbose=1)
        all_logits.append(logits)

    logits = tf.concat(all_logits, 0)
    probs = tf.nn.softmax(logits)

    final_probs = probs[-1]
    print(final_probs)
  except:
    print('did not work')

  saved_model_dir = f'Models/MoViNet/models/{model_id}/{name}'
  input_shape = [1, 1 if model_id[2:] == '_stream' else frames, resolution, resolution, 3]
  input_shape_concrete = [1 if s is None else s for s in input_shape]
  print(input_shape)
  print(input_shape_concrete)
  print(model_id[2:] == '_stream')

  export_saved_model.export_saved_model(
    model=model,
    input_shape=input_shape,
    export_path=saved_model_dir,
    causal=model_id[2:] == '_stream',
    bundle_input_init_states_fn=False)
  
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  tflite_model = converter.convert()

  with open(f'{saved_model_dir}.tflite', 'wb') as f:
    f.write(tflite_model)

  # Create the interpreter and signature runner
  interpreter = tf.lite.Interpreter(model_path=f'{saved_model_dir}.tflite')
  runner = interpreter.get_signature_runner()

  init_states = {
      name: tf.zeros(x['shape'], dtype=x['dtype'])
      for name, x in runner.get_input_details().items()
  }
  del init_states['image']

  states = init_states
  for frames, label in test_dataset.take(1):
    for clip in frames[0]:
      # Input shape: [1, 1, 172, 172, 3]
      outputs = runner(**states, image=clip)
      logits = outputs.pop('logits')[0]
      states = outputs

  probs = tf.nn.softmax(logits)
  print(probs)

  frames, label = list(test_dataset.take(1))[0]

  # converter = tf.lite.TFLiteConverter.from_saved_model(f'{saved_model_dir}') # path to the SavedModel directory
  # converter.target_spec.supported_ops = [
  #   tf.lite.OpsSet.TFLITE_BUILTINS,
  #   tf.lite.OpsSet.SELECT_TF_OPS
  # ]
  # tflite_model = converter.convert()

  # # Save the model.
  # with open(f'{name}.tflite', 'wb') as f:
  #   f.write(tflite_model)


  # model.build(input_shape_concrete)
  # _ = model(tf.ones(input_shape_concrete))
  # tf_keras.models.save_model(model, saved_model_dir)
  # tf.saved_model.save(model,f'{saved_model_dir}_1')

  # converter = tf.lite.TFLiteConverter.from_saved_model(f'{saved_model_dir}_1')
  # tflite_model = converter.convert()
  
  # with open(f'{saved_model_dir}_lite/{name}.tflite', 'wb') as f:
  #   f.write(tflite_model)
  # model = build_model_inference(model_id, batch_size=batch_size, num_frames=frames, resolution=resolution,name=name, num_classes=num_classes, activation=activation )



  # Convert to saved model





  # # input_shape_concrete = [1 if s is None else s for s in None]
  # # _ = model(tf.ones(input_shape_concrete))
  # tf_keras.models.save_model(model, f'{saved_model_dir}_2')
  # # tf.saved_model.save(model, saved_model_dir)





  
  # converter = tf.lite.TFLiteConverter.from_saved_model(f'{saved_model_dir}_1')
  # tflite_model = converter.convert()
  
  # with open(f'{saved_model_dir}_lite/{name}.tflite', 'wb') as f:
  #   f.write(tflite_model)
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
  parser = argparse.ArgumentParser(description="Run Movinet")
  parser.add_argument("-b", "--batch_size", help="batch size to use for training", default=4, required=False, type=int)
  parser.add_argument("-l", "--learning_rate", help="Learning rate", default=0.01, required=False, type=float)
  parser.add_argument("-d", "--dropout_rate", help="Stochastic depth dropout rate", required=False, default=0.0, type=float)
  parser.add_argument("-v", "--version", help="Version of movinet, a0, a1, a2", required=False, default='a0', type=str)
  parser.add_argument("-s", "--stream", help="Is the model a stream model", required=False, default=1, type=int)
  parser.add_argument("-e", "--epochs", help="Epocs, 1, 2, 20, 100", required=False, default=20, type=int)

  

  tf.random.set_seed(123)

  args = parser.parse_args()
  main(args)