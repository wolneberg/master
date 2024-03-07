import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import tensorflow as tf

def plot_history(history, name):
  """
    Plotting training and validation learning curves.

    Args:
      history: model history with all the metric measures
  """
  fig, (ax1, ax2) = plt.subplots(2)

  fig.set_size_inches(18.5, 10.5)

  # Plot loss
  ax1.set_title('Loss')
  ax1.plot(history.history['loss'], label = 'train')
  ax1.plot(history.history['val_loss'], label = 'test')
  ax1.set_ylabel('Loss')

  # Determine upper bound of y-axis
  max_loss = max(history.history['loss'] + history.history['val_loss'])

  ax1.set_ylim([0, np.ceil(max_loss)])
  ax1.set_xlabel('Epoch')
  ax1.legend(['Train', 'Validation']) 

  # Plot accuracy
  ax2.set_title('Accuracy')
  ax2.plot(history.history['accuracy'],  label = 'train')
  ax2.plot(history.history['val_accuracy'], label = 'test')
  ax2.set_ylabel('Accuracy')
  ax2.set_ylim([0, 1])
  ax2.set_xlabel('Epoch')
  ax2.legend(['Train', 'Validation'])

  plt.savefig(f'Models/MoViNet/data/results/{name}.png')

# def plot_confusion_matrix(name, actual, predicted, labels, ds_type):
  # cm = tf.math.confusion_matrix(actual, predicted)
  # ax = sns.heatmap(cm, annot=True, fmt='g')
  # sns.set(rc={'figure.figsize':(12, 12)})
  # sns.set(font_scale=1.4)
  # ax.set_title('Confusion matrix of action recognition for ' + ds_type)
  # ax.set_xlabel('Predicted Action')
  # ax.set_ylabel('Actual Action')
  # plt.xticks(rotation=90)
  # plt.yticks(rotation=0)
  # ax.xaxis.set_ticklabels(labels)
  # ax.yaxis.set_ticklabels(labels)

  # plt.savefig(f'Models/MoViNet/data/results/{name}.png')