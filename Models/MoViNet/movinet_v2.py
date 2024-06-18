import os
import tensorflow as tf

from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

"""Build MoViNet model with pretrained weights on Kinetics dataset"""
def build_classifier(model_id, batch_size, num_frames, resolution, num_classes, unFreezLayers, dropout_rate, stochastic_depth_drop_rate=0):
  model = None
  use_positional_encoding = model_id[:2] in {'a3', 'a4', 'a5'}

  if model_id[2:] == '_base': 
    backbone = movinet.Movinet(
        model_id=model_id[:2],
        causal=False,
        stochastic_depth_drop_rate = stochastic_depth_drop_rate,
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
  for layer in backbone.layers[:-unFreezLayers]:
      layer.trainable = False

  # Set num_classes=600 to load the pre-trained weights on kinetics from the original model
  model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600, output_states=model_id[2:] == '_stream')
  # Load pre-trained weights on Kinetics dataset
  checkpoint_dir = f'Models/MoViNet/Backbone/movinet_{model_id}'
  checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
  checkpoint = tf.train.Checkpoint(model=model)
  status = checkpoint.restore(checkpoint_path)
  status.assert_existing_objects_matched()

  model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=num_classes, dropout_rate=dropout_rate)
  model.build([batch_size, num_frames, resolution, resolution, 3])
  return model

"""Build MoViNet model for inference for the stream version"""
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
    use_external_states=True, #This need to be True for inference mode, but False for training mode
  )

  model = movinet_model.MovinetClassifier(
      backbone,
      num_classes=100,
      output_states=True)
  inputs = tf.ones([1, num_frames, resolution, resolution, 3])
  model.build(inputs)

  print('Load checkpoint from the training')
  checkpoint_path = f"Models/MoViNet/checkpoints/hyperparameter/{model_id}/{name}/cp.ckpt"
  model.load_weights(checkpoint_path).expect_partial()
  return model

"""Compile MoViNet with loss and optimizers"""
def compile(model, len_train, batch_size, epochs, optimizer, learning_rate=0.01, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0):
  loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  if optimizer == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
  elif optimizer == 'rmsprop':
    train_steps = len_train // batch_size
    total_train_steps = train_steps * epochs
    initial_learning_rate = learning_rate
    learning_rate = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps=total_train_steps)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate, rho=rho, momentum=momentum, epsilon=epsilon, clipnorm=clipnorm)
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate)

  model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])
  del loss_obj, optimizer, learning_rate  

"""Train MoViNet model"""
def train(model, train_ds, val_ds, epochs, name, model_id):
  print("Training a movinet model...")
  print(tf.config.list_physical_devices('GPU'))
  checkpoint_path = f"Models/MoViNet/checkpoints/hyperparameter/{model_id}/{name}/cp.ckpt"
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






