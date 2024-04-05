from WLASL.extraction import get_glosses, get_video_subset
from WLASL.preprocess import format_dataset, get_less_glosses, get_frame_set
# from Models.V3DCNN.v3dcnn import train
from Models.MoViNet.movinet_v2 import compile, build_classifier, train
from utils.plot import plot_history
import tensorflow as tf, tf_keras
import datetime
from official.projects.movinet.tools import export_saved_model

versions = {'a0': ['a0_base'],'a1': ['a1_base'], 'a2': ['a2_base', 'a2_stream']}

model = 'movinet'
model_id = versions.get('a1')[0]
name= f'movinet_{model_id}_{datetime.datetime.now().strftime("%d%m%Y-%H%M%S")}'

batch_size = 4
num_classes = 100
frames = 20
frame_step = 2
resolution = 172
epochs = 15
activation = 'softmax'
num_unfreeze_layers = 3

optimizer = 'rmsprop'
learning_rate = 0.01
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
print(f'Number of unfreezed layers: {num_unfreeze_layers}')
print(f'batch size: {batch_size}')
print(f'Frames: {frames}, frame step: {frame_step}, resolution: {resolution}')
print(f'Initial learning rate: {learning_rate}, Rho: {rho}, Momentum: {momentum}, Epsilon: {epsilon}, clipnorm: {clipnorm} ')
# print("Avere pooling type er 2d")
print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

print(f'{name}, epochs: {epochs}, classes: {num_classes}, batch size: {batch_size}')


print('Getting train videos...\n')
train_videos = get_video_subset(f'nslt_{num_classes}', 'train')
print('\nGetting validation videos...\n')
val_videos = get_video_subset(f'nslt_{num_classes}', 'val')
print('\nGetting test videos...\n')
test_videos = get_video_subset(f'nslt_{num_classes}', 'test')
print('Getting glosses...\n')
glosses = get_glosses()

train_set = get_frame_set(train_videos, glosses, frames, resolution, frame_step)
val_set = get_frame_set(val_videos, glosses, frames, resolution, frame_step)
# glosses = get_less_glosses(train_set)
test_set = get_frame_set(test_videos, glosses, frames, resolution, frame_step)

print('formatting train...')
train_dataset = format_dataset(train_set, glosses, over=False)
print('formatting val...')
val_dataset = format_dataset(val_set, glosses, over=False)
test_dataset = format_dataset(test_set, glosses, over=False)

print('batching...')
train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(1)

for element, label in test_dataset:
  test_input = element
  break

print('element: ')
print(element)

print('Training...\n')
model = build_classifier(model_id, batch_size, frames, resolution, num_classes, num_unfreeze_layers, activation)
compile(model, len(train_set), batch_size, epochs, optimizer)

result = train(model, train_ds=train_dataset, val_ds=val_dataset, epochs=epochs)
plot_history(result, name)

print("---------------------")
print("Inference 1...")
print("---------------------")



output = model(test_input, training=False)
print(output)
# # tf.saved_model.save(model, f'Models/MoViNet/models/{name}')
# # converter = tf.lite.TFLiteConverter.from_keras_model(model)
# # tflite_model = converter.convert()
# # with open('Models/MoViNet/lite/model.tflite', 'wb') as f:
# #   f.write(tflite_model)
# # onnx_model, _ = tf2onnx.convert.from_keras(model, [batch_size, frames, resolution, resolution, 3], opset=13)
# # onnx.save(onnx_model, "dst/path/model.onnx")
# # new_model = tf.keras.models.load_model('my_model.keras')

print("---------------------")
print("Saving...")
print("---------------------")

# model.save(f'Models/MoViNet/models/{name}')
# tf.saved_model.save(model, f'Models/MoViNet/models/{name}')
# tf_keras.models.save_model(model, f'Models/MoViNet/models/{name}')

input_shape = [1, frames, resolution, resolution, 3]
export_saved_model.export_saved_model(model=model, input_shape=input_shape,export_path=f'Models/MoViNet/models/{name}')


print("---------------------")
print("converting")
print("---------------------")

model = tf.saved_model.load(f"Models/MoViNet/models/{name}") 
concrete_func = model.signatures[ tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY] 
concrete_func.inputs[0].set_shape([1, frames, resolution, resolution, 3])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func]) 
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS,
  tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()

# # converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
# # tflite_model = converter.convert()
# # open('Models/MoViNet/lite/model.tflite', 'wb').write(tflite_model)

# Convert the model
# converter = tf.lite.TFLiteConverter.from_saved_model(f"Models/MoViNet/models/{name}") # path to the SavedModel directory
# converter.target_spec.supported_ops = [
#   tf.lite.OpsSet.TFLITE_BUILTINS,
#   tf.lite.OpsSet.SELECT_TF_OPS
# ]
# tflite_model = converter.convert()




# Save the model.
with open(f'Models/MoViNet/lite/{name}.tflite', 'wb') as f:
  f.write(tflite_model)

print("---------------------")
print("Inference 2...")
print("---------------------")

loaded_model = tf.saved_model.load(f'Models/MoViNet/models/{name}')

sig = loaded_model.signatures['serving_default']
print(sig.pretty_printed_signature())
logits = sig(image=test_input)
print(logits)