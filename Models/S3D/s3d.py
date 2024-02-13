import numpy as np
import pandas as pd
import tensorflow as tf
import random
import cv2

import os
from official.projects.s3d.modeling import s3d

def build_backbone():
  backbone = s3d.S3D()
  print(backbone.summary())

