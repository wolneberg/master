from WLASL.extraction import get_glosses, get_video_subset
from WLASL.preprocess import format_dataset, get_frame_set
import tensorflow as tf
import keras

import sys
print(sys.version)
# import keras
# from tensorflow.keras.models import load_model
num_classes = 100
frames = 20
frame_step = 2
resolution = 172
test_videos = get_video_subset(f'nslt_{num_classes}', 'test')
glosses = get_glosses()

test_set = get_frame_set(test_videos, glosses, frames, resolution, frame_step)

test_dataset = format_dataset(test_set, glosses, over=False)

test_dataset = test_dataset.batch(1)


# new_model = tf.keras.models.load_model('movinet_freez10_3.keras')
model = tf.saved_model.load('Models/MoViNet/models/movinet_freez10_3')
# model = tf.saved_model.load('saved_model_keras_dir')
# @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None, None, 3), dtype=tf.float32)])
# input_tensor = tf.random.uniform(shape=[frames,3,resolution,resolution])
# out1, out2 = model(input_tensor) 
# print(new_model.signatures)

# new_model
# model = keras.models.load_model("movinet.keras")
# model = tf.keras.models.load_model("saved_model_keras_dir")
# model1 = tf.keras.models.load_model("movinet.keras")
# new_model = tf.keras.saving.load_model("movinet_freez10_3.keras")

# model.evaluate(test_dataset)
# prediction = model(test_set.iloc[1]['frames'])
# prediction = model(test_dataset.take(1))
# print(prediction)
# def serving(test):
#   return model(test)
# print(serving(test_dataset)[:10])
# print(test_set.iloc[:10]['gloss'])

# Convert the model
# converter = tf.lite.TFLiteConverter.from_saved_model(f'Models/MoViNet/models/movinet1') # path to the SavedModel directory
# tflite_model = converter.convert()

# # Save the model.
# with open('model.tflite', 'wb') as f:
#     f.write(tflite_model)

# model1.summary()

# try:
  # Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('Models/MoViNet/models/movinet_freez10_3') # path to the SavedModel directory
tflite_model = converter.convert()

  # Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
# except:
  # print('Was not able to save lite version from saved models')