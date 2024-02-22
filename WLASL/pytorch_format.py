from typing import Optional, Callable
from torchvision.datasets.vision import VisionDataset


class WLASLDataset(VisionDataset):

  def __init__(self,path, video_subset, label_df, transform: Optional[Callable] = None,):
    self.path = path
    self.video_subset = video_subset
    self.label_df = label_df
    self.transform = transform

  def __len__(self):
    return len(self.video_subset)
  
  def __getitem__(self, index):
    video_id = self.video_subset[index]
    video_path = self.path + f'{video_id:05}' + '.mp4'
    label = self.label_df[self.label_df['video_id'] == f'{video_id:05}'].iloc[0]['gloss']
    return (video_path, label)