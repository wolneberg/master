import os

from torchvision.datasets.vision import VisionDataset

def make_dataset(path, video_subset, label_df):
  '''
  Create a dataset from a given subset.
  Return a list of tuples (video_path, label)
  '''
  dataset = []
  for video_id in video_subset:
    video_path = path + f'{video_id:05}' + '.mp4'
    # Check if the videopath exists
    if os.path.isfile(video_path):
      label = label_df[label_df['video_id'] == f'{video_id:05}'].iloc[0]['gloss_id']
      label = int(label)
      dataset.append((video_path, label))
  return dataset

class WLASLDataset(VisionDataset):

  def __init__(self, dataset, transform = None):
    self.dataset = dataset
    self.transform = transform

  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self, index):
    video = self.dataset[index][0]
    label = self.dataset[index][1]
    if self.transform:
      video = self.transform(video)
    return video, label