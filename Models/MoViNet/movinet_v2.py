import os
import tensorflow as tf

from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model


def build_classifier(model_id, batch_size, num_frames, resolution, num_classes, activation, freeze_backbone=False, stochastic_depth_drop_rate=0):
  """Builds a classifier on top of a backbone model."""
  # tf.keras.backend.clear_session()
  model = None
  use_positional_encoding = model_id[:2] in {'a3', 'a4', 'a5'}

  if model_id[2:] == '_base': 
    backbone = movinet.Movinet(
        model_id=model_id[:2],
        causal=False,
        conv_type='2plus1d',
        se_type='2plus3d', 
        stochastic_depth_drop_rate = stochastic_depth_drop_rate,
        # use_external_states=False,
  )
  else:
    backbone = movinet.Movinet(
      model_id=model_id[:2], 
      causal=True,
      conv_type='2plus1d',
      se_type='2plus3d', 
      activation='hard_swish',
      gating_activation='hard_sigmoid', 
      use_positional_encoding = use_positional_encoding,
      stochastic_depth_drop_rate = stochastic_depth_drop_rate,
      use_external_states=False,
    )
  
  backbone.trainable = True
  # for layer in backbone.layers[:-unFreez]:
  #       layer.trainable = False

  # Set num_classes=600 to load the pre-trained weights from the original model
  model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600, output_states=model_id[2:] == '_stream')
  inputs = tf.ones([1, 13, 172, 172, 3])
  model.build(inputs)
  # model.build([None, None, None, None, 3])

  # Load pre-trained weights
  checkpoint_dir = f'Models/MoViNet/Backbone/movinet_{model_id}'
  checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
  checkpoint = tf.train.Checkpoint(model=model)
  status = checkpoint.restore(checkpoint_path)
  status.assert_existing_objects_matched()


  model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=100, output_states=model_id[2:] == 'stream')


  # # print(x)
  # output = tf.keras.layers.Dense(units=num_classes, activation=activation)(model.layers[-1].output)
  # model = tf.keras.Model(inputs=model.inputs, outputs=output)
  # model = tf.keras.Sequential([
  #    model,
  #    tf.keras.layers.Dense(num_classes)
  # ])
  model.build([batch_size, num_frames, resolution, resolution, 3])
  # del output, backbone
  # del checkpoint, checkpoint_dir, checkpoint_path, status
  if freeze_backbone:
    for layer in model.layers[:-1]:
        layer.trainable = False
    model.layers[-1].trainable = True
  return model

# with strategy.scope():

def build_model_inference(model_id, num_frames, resolution, name):
    
  use_positional_encoding = model_id[:2] in {'a3', 'a4', 'a5'}

  backbone = movinet.Movinet(
    model_id=model_id[:2],
    causal=True,
    conv_type='2plus1d',
    se_type='2plus3d',
    activation='hard_swish',
    gating_activation='hard_sigmoid',
    use_positional_encoding=use_positional_encoding,
    use_external_states=True,
  )

  model = movinet_model.MovinetClassifier(
      backbone,
      num_classes=100,
      output_states=True)
  inputs = tf.ones([1, num_frames, resolution, resolution, 3])
  model.build(inputs)

  print('load checkpoint')
  checkpoint_path = f"Models/MoViNet/checkpoints/hyperparameter/{model_id}/{name}/cp.ckpt"
  # print(tf.train.latest_checkpoint(checkpoint_path))
  model.load_weights(checkpoint_path)

  return model


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


def train(model, train_ds, val_ds, epochs, name, model_id):
  print("Training a movinet model...")
  print(model.summary())
  print(tf.config.list_physical_devices('GPU'))
  print(tf.config.list_physical_devices())
  checkpoint_path = f"Models/MoViNet/checkpoints/hyperparameter/{model_id}/{name}/cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)

  # Create a callback that saves the model's weights
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  save_best_only=True,
                                                  verbose=1)
  results = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2, callbacks=[cp_callback])
  print('done traininer')
  model.load_weights(checkpoint_path)
  return results

# def evaluate(model, test_ds):
#    print("Evaluate a movinet Mode...")
#    loss, acc = model.evaluate(test_ds, verbose=1)
#    print("Model, accuracy: {:5.2f}%".format(100 * acc))
#    return loss, acc

# def get_actual_predicted_labels(model, dataset):
#   """
#     Create a list of actual ground truth values and the predictions from the model.

#     Args:
#       dataset: An iterable data structure, such as a TensorFlow Dataset, with features and labels.

#     Return:
#       Ground truth and predicted values for a particular dataset.
#   """
#   actual = [labels for _, labels in dataset.unbatch()]
#   predicted = model.predict(dataset)

#   actual = tf.stack(actual, axis=0)
#   predicted = tf.concat(predicted, axis=0)
#   predicted = tf.argmax(predicted, axis=1)

#   return actual, predicted






