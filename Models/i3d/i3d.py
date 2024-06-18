import os
import tensorflow as tf
from Models.i3d.I3D_model.i3d_inception import Inception_Inflated3d, conv3d_bn
from keras import backend as backend


"""Builds the I3D model from Oana Ignat implementation with imagenet and kinetics weights"""
def build_classifier(num_frames, resolution, num_classes, unFreezLayers, dropout_rate):
  """Builds a classifier on top of a backbone model."""
  model = Inception_Inflated3d(
                include_top=False,
                endpoint_logit=False,
                weights='rgb_imagenet_and_kinetics',
                input_shape=(num_frames, resolution, resolution, 3),
                classes=100)
  x = model.layers[-1].output
  x = tf.keras.layers.Dropout(dropout_rate)(x)
  x = conv3d_bn(x, num_classes, 1, 1, 1, padding='same', 
          use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')
  num_frames_remaining = int(x.shape[1])
  x = tf.keras.layers.Reshape((num_frames_remaining, num_classes))(x)
  x = tf.keras.layers.Lambda(lambda x: backend.mean(x, axis=1, keepdims=False),
              output_shape=lambda s: (s[0], s[2]))(x)
  x = tf.keras.layers.Activation('softmax', name='prediction')(x)
  model = tf.keras.Model(inputs=model.inputs, outputs=x)
  
  model.trainable = True
  for layer in model.layers[:-unFreezLayers]:
      layer.trainable = False
  print(model.summary())

  return model

"""Compile I3D model with loss and optimizer"""
def compile(model, learning_rate=0.01):
  loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
  optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
  model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])
  del loss_obj, optimizer, learning_rate
  
"""Training the I3D model"""
def train(model, train_ds, val_ds, epochs, name):
  print("Training a movinet model...")
  print(tf.config.list_physical_devices('GPU'))
  checkpoint_path = f"Models/i3d/checkpoints/{name}/cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)

  # Create a callback that saves the model's weights
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  save_best_only=True,
                                                  monitor='val_accuracy',
                                                  verbose=1)
  results = model.fit(train_ds, 
                      validation_data=val_ds, 
                      epochs=epochs, 
                      verbose=2, 
                      callbacks=[cp_callback])
  print('Done Training')
  model.load_weights(checkpoint_path)
  return results
