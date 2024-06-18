# Sign Language Recognition
This contains different deep learning models used for sign language recognition (SLR) which was used for the master thesis of Ingrid Marie Wølneberg and Helene Amlie. It contains the models [MoViNet](https://arxiv.org/abs/2103.11511), [I3D](https://arxiv.org/abs/1705.07750), [S3D](https://arxiv.org/abs/1711.11248v3), and [X3D](https://arxiv.org/abs/2004.04730) for the [WLASL dataset](https://arxiv.org/abs/1910.11006).

TensorFlow and PyTorch was used for the models, and they were converted to TensorFlow Lite and ONNX for running on a mobile application.

We hyperparameter tuned MoViNet and I3D using Keras [HyperBand](https://keras.io/api/keras_tuner/tuners/hyperband/) and [Bayesian](https://keras.io/api/keras_tuner/tuners/bayesian/#bayesianoptimization-class) hyperparameter optimizers.

Tutorials and code used for each model: 
### I3D
- The repository by [Oana Ignat](https://github.com/OanaIgnat/I3D_Keras) was the implementation used for I3D

### MoViNet
- [MoViNet tutorial](https://github.com/tensorflow/models/blob/master/official/projects/movinet/movinet_tutorial.ipynb) by TensorFlow
- [Movinet Steaming Model Tutorial](https://github.com/tensorflow/models/blob/master/official/projects/movinet/movinet_streaming_model_training_and_inference.ipynb) by TensorFlow

### S3D
- [Video S3D](https://pytorch.org/vision/main/models/video_s3d.html) by PyTorch

### X3D
- [X3D networks pretrained on the Kinetics 400 dataset](https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/) by FAIR PyTorchVideo


# Results
We only hyperparameter tuned MoViNet and I3D as those showed the most promizing results.
Result on WLASL100 are shown in the table below. 
Resolution $172^2$ was used for MoViNets, and $224^2$ was used for I3D. All model are trained on 20 frames.

| Model | Top 1 | Top 2 | Loss |
|----------|----------|----------| -----------|
| I3D | 48.45 | 76.35  | 2.2436 |
| MoViNet-a0 | 53.49 | 82.17 | 2.5375 |
| MoViNet-a1 | 59.30 | 85.66 | 2.4041 |
| MoViNet-a2 | 61.63 | 87.21 | 1.9817 |
| MoViNet-a3 | 63.18 | 86.43 | 1.9503 |
| MoViNet-a4 | 60.47 | 87.60 | 1.9429 |
| MoViNet-a5 | 60.85 | 86.43 | 2.2652 |


# Filestucture

### Models
- This folder contains all the different models.
- Each folder for the model contains a model file containing code to create, compile, and train the model. 
- The output files, model, checkpoints, and plots will also be saved here.
- For I3D, the implementation of I3D from Oana Ignat is added to the folder

### WLASL
- This folder contains the pre-processing of the WLASL videos, with cropping, frame extractions, and transformation. 
- The videos are put in this folder if you want to test our code.

### Utils
- The utils folder contains a few helper functions. 
    - Validation loss and accuracy plots and top k accuracy function.

### Main 
- In the main folder, all the training files for each model with its corresponding slurm and requirements file. 
- Some of the training files take in some arguments (epochs, batch size, frees layers, dropout rate, etc) that can be changed to test different values. These can be changed in the slurm file.

