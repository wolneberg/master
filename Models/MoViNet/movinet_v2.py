import tensorflow as tf

from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

try:
  print("Physical devices: ",tf.config.list_physical_devices('GPU'))
  print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
except:
  print('gikk ikke å printe')


def build_classifier(model_id, batch_size, num_frames, resolution, num_classes, unFreez, activation, freeze_backbone=False):
  """Builds a classifier on top of a backbone model."""
  # tf.keras.backend.clear_session()
  
  backbone = movinet.Movinet(model_id=model_id[:2])
  backbone.trainable = True
  for layer in backbone.layers[:-unFreez]:
        layer.trainable = False

  # Set num_classes=600 to load the pre-trained weights from the original model
  model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
  model.build([None, None, None, None, 3])

  # Load pre-trained weights
  checkpoint_dir = f'Models/MoViNet/data/movinet_{model_id}'
  checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
  checkpoint = tf.train.Checkpoint(model=model)
  status = checkpoint.restore(checkpoint_path)
  status.assert_existing_objects_matched()

  x = model.layers[-1].output
  # print(x)
  output = tf.keras.layers.Dense(units=num_classes, activation=activation)(x)
  for element in model.inputs:
     print(element)
     break

  model = tf.keras.Model(inputs=model.inputs, outputs=output)
  model.build([batch_size, num_frames, resolution, resolution, 3])
  if freeze_backbone:
    for layer in model.layers[:-1]:
        layer.trainable = False
    model.layers[-1].trainable = True
  return model



def compile(model, len_train, batch_size, epochs, optimizer, learning_rate=0.01, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0):

  loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()

  if optimizer == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
  elif optimizer == 'rmsprop':
    train_steps = len_train // batch_size
    total_train_steps = train_steps * epochs
    initial_learning_rate = learning_rate
    learning_rate = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps=total_train_steps,)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate, rho=rho, momentum=momentum, epsilon=epsilon, clipnorm=clipnorm)

  model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])



def train(model, train_ds, val_ds, epochs):
  print("Training a movinet model...")

  results = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)
  return results

# def evaluate(model, test_ds):
#    print("Evaluate a movinet Mode...")
#    loss, acc = model.evaluate(test_ds, verbose=2)
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






