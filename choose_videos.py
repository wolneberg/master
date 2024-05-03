import os
import random
import shutil

path_to_all_test_videos = "/Users/ingridmariewolneberg/Desktop/test_vid/"
path_to_save = "/Users/ingridmariewolneberg/Desktop/MobileTestVideos/"

test_videos = os.listdir(path_to_all_test_videos)

if not os.path.exists(path_to_save):
  os.mkdir(path_to_save)

random.seed(42)
rand_index = random.randint(0,len(test_videos)-1)
file_name = test_videos[rand_index]
shutil.copyfile(path_to_all_test_videos+file_name, path_to_save+file_name)
for _ in range(39):
  while os.path.exists(path_to_save+file_name):
    rand_index = random.randint(0,len(test_videos)-1)
    file_name = test_videos[rand_index]
  shutil.copyfile(path_to_all_test_videos+file_name, path_to_save+file_name)