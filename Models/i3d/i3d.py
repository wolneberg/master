import os
import tensorflow as tf
import tensorflow_hub as hub
from Models.i3d.I3D_model.i3d_inception import Inception_Inflated3d, conv3d_bn
from keras import backend as backend

# from Models.i3d.kinetics_i3d.i3d import InceptionI3d


def build_classifier(batch_size, num_frames, resolution, num_classes, unFreezLayers):
  """Builds a classifier on top of a backbone model."""
  model = Inception_Inflated3d(
                include_top=False,
                endpoint_logit=False,
                dropout_prob = 0.0, 
                weights='rgb_imagenet_and_kinetics',
                input_shape=(num_frames, resolution, resolution, 3),
                classes=num_classes)
  x = model.layers[-1].output
  x = tf.keras.layers.Dropout(0.1)(x)

  x = conv3d_bn(x, num_classes, 1, 1, 1, padding='same', 
          use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

  num_frames_remaining = int(x.shape[1])
  x = tf.keras.layers.Reshape((num_frames_remaining, num_classes))(x)

  # logits (raw scores for each class)
  x = tf.keras.layers.Lambda(lambda x: backend.mean(x, axis=1, keepdims=False),
              output_shape=lambda s: (s[0], s[2]))(x)

  x = tf.keras.layers.Activation('softmax', name='prediction')(x)
  # x = tf.keras.layers.Dense(units=num_classes, activation='softmax')(x)
  model = tf.keras.Model(inputs=model.inputs, outputs=x)
  # model = tf.keras.Sequential([
  #    model,
  #    tf.keras.layers.Dense(num_classes)
  # ])
  
  model.trainable = True
  for layer in model.layers[:-unFreezLayers]:
      layer.trainable = False
  # model = InceptionI3d(num_classes=100, final_endpoint='Predictions')
  
  # print(model.summary())

  return model

# with strategy.scope():

def compile(model, len_train, batch_size, epochs, optimizer, learning_rate=0.01, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0):
  loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


  if optimizer == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
  elif optimizer == 'rmsprop':
    train_steps = len_train // batch_size
    total_train_steps = train_steps * epochs
    initial_learning_rate = learning_rate
    learning_rate = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps=total_train_steps)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate, rho=rho, momentum=momentum, epsilon=epsilon, clipnorm=clipnorm)

  model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])
  del loss_obj, optimizer, learning_rate

def build_model(hp):
  # stochastic_depth_drop_rate = hp.Float('stochastic_depth_drop_rate', min_value=0.0, max_value=0.5, step=0.1)
  dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
  model = Inception_Inflated3d(
                include_top=False,
                endpoint_logit=False,
                dropout_prob = dropout_rate, 
                weights='rgb_imagenet_and_kinetics',
                input_shape=(20, 224, 224, 3),
                classes=100)
  output = tf.keras.layers.Dense(units=100, activation='softmax')(model.layers[-1].output)
  model = tf.keras.Model(inputs=model.inputs, outputs=output)
  # model = tf.keras.Sequential([
  #    model,
  #    tf.keras.layers.Dense(num_classes)
  # ])
  # print(len(model.layers))
  unfreezLayers = hp.Int('freez_layers', min_value=0, max_value=len(model.layers)-50, step=10)

  model.trainable = True
  for layer in model.layers[:-unfreezLayers]:
      layer.trainable = False

  loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

  learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')

  optimizer  = tf.keras.optimizers.RMSprop(learning_rate)
  model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])
  
  return model
  

def train(model, train_ds, val_ds, epochs, name):
  print("Training a movinet model...")
  # print(model.summary())
  print(tf.config.list_physical_devices('GPU'))
  # print(tf.config.list_physical_devices())
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
                      # validation_freq=1,
                      verbose=2, 
                      callbacks=[cp_callback])
  print('done traininer')
  model.load_weights(checkpoint_path)
  return results
