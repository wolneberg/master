import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils

mobile = tf.keras.applications.mobilenet.MobileNet()

def prepare_image(file):
    img_path = 'data/MobileNet-samples/'
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


preprocessed_image = prepare_image('1.PNG')
predictions = mobile.predict(preprocessed_image)

results = imagenet_utils.decode_predictions(predictions)

print(results)