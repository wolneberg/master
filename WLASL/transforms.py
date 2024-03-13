import torch
import torchvision
import numpy as np
import PIL
import collections 
import random
import cv2

# from https://github.com/YuxinZhaozyx/pytorch-VideoDataset/blob/master/transforms.py 

__all__ = ['FramesFromVideoPath','VideoFilePathToTensor', 'VideoResize', 'VideoRandomCrop', 'VideoCenterCrop', 'VideoRandomHorizontalFlip', 
            'VideoRandomVerticalFlip', 'VideoGrayscale']

class FramesFromVideoPath(object):
    '''
        Prøvde å lage lignende som for tensorflow koden vår, men funka dårlig
    '''

    def __init__(self, num_frames, output_size=(256,256), frame_step=5, max_length=100):
        self.num_frames = num_frames
        self.output_size = output_size
        self.frame_step = frame_step
        self.max_length = max_length
        self.channels = 3

    def __call__(self, path):
        # open video file
        cap = cv2.VideoCapture(str(path))
        assert(cap.isOpened())
        video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # init empty output frames (C x L x H x W)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        need_length = 1 + (self.num_frames - 1) * self.frame_step
    
        if need_length > video_length:
            start = 0
        else:
            max_start = video_length - need_length
            start = random.randint(0, self.max_length)

        frames = torch.FloatTensor(self.channels, self.max_length+1, height, width)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        # ret is a boolean indicating whether read was successful, frame is the image itself
        ret, frame = cap.read()
        # successfully read frame
        # BGR to RGB
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            frame = torch.from_numpy(frame)
            # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames[:, start, :, :] = frame.float()
        else:
            frames[:, start:, :, :] = 0
        for _ in range(self.num_frames-1):
            for _ in range(self.frame_step):
                # read frame
                ret, frame = cap.read()
            if ret:
                # successfully read frame
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
                frame = torch.from_numpy(frame)
                # (H x W x C) to (C x H x W)
                frame = frame.permute(2, 0, 1)
                frames[:, _, :, :] = frame.float()
            else:
                # reach the end of the video
                # fill the rest frames with 0.0
                frames[:, _:, :, :] = 0

        frames /= 255
        cap.release()
        return frames

class VideoFilePathToTensor(object):
    """ load video at given file path to torch.Tensor (C x L x H x W, C = 3) 
        It can be composed with torchvision.transforms.Compose().
        
    Args:
        max_len (int): Maximum output time depth (L <= max_len). Default is None.
            If it is set to None, it will output all frames. 
        fps (int): sample frame per seconds. It must lower than or equal the origin video fps.
            Default is None. 
        padding_mode (str): Type of padding. Default to None. Only available when max_len is not None.
            - None: won't padding, video length is variable.
            - 'zero': padding the rest empty frames to zeros.
            - 'last': padding the rest empty frames to the last frame.
    """

    def __init__(self, max_len=None, fps=None, padding_mode=None):
        self.max_len = max_len
        self.fps = fps
        assert padding_mode in (None, 'zero', 'last')
        self.padding_mode = padding_mode
        self.channels = 3  # only available to read 3 channels video

    def __call__(self, path):
        """
        Args:
            path (str): path of video file.
            
        Returns:
            torch.Tensor: Video Tensor (C x L x H x W)
        """
        
        # open video file
        cap = cv2.VideoCapture(str(path))
        assert(cap.isOpened())
       
        # calculate sample_factor to reset fps
        sample_factor = 1
        if self.fps:
            old_fps = cap.get(cv2.CAP_PROP_FPS)  # fps of video
            print(old_fps)
            sample_factor = int(old_fps / self.fps)
            assert(sample_factor >= 1)
        
        # init empty output frames (C x L x H x W)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        time_len = None
        if self.max_len:
            # time length has upper bound
            if self.padding_mode:
                # padding all video to the same time length
                time_len = self.max_len
            else:
                # video have variable time length
                time_len = min(int(num_frames / sample_factor), self.max_len)
        else:
            # time length is unlimited
            time_len = int(num_frames / sample_factor)

        frames = torch.FloatTensor(self.channels, time_len, height, width)

        for index in range(time_len):
            frame_index = sample_factor * index

            # read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                # successfully read frame
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
                frame = torch.from_numpy(frame)
                # (H x W x C) to (C x H x W)
                frame = frame.permute(2, 0, 1)
                frames[:, index, :, :] = frame.float()
            else:
                # reach the end of the video
                if self.padding_mode == 'zero':
                    # fill the rest frames with 0.0
                    frames[:, index:, :, :] = 0
                elif self.padding_mode == 'last':
                    # fill the rest frames with the last frame
                    assert(index > 0)
                    frames[:, index:, :, :] = frames[:, index-1, :, :].view(self.channels, 1, height, width)
                break

        frames /= 255
        cap.release()
        return frames

class VideoResize(object):
    """ resize video tensor (C x L x H x W) to (C x L x h x w) 
    
    Args:
        size (sequence): Desired output size. size is a sequence like (H, W),
            output size will matched to this.
        interpolation (int, optional): Desired interpolation. Default is 'PIL.Image.BILINEAR'
    """

    def __init__(self, size, interpolation=PIL.Image.BILINEAR):
        assert isinstance(size, collections.abc.Iterable) and len(size) == 2
        self.size = size
        self.interpolation = interpolation

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video to be scaled (C x L x H x W)
        
        Returns:
            torch.Tensor: Rescaled video (C x L x h x w)
        """

        h, w = self.size
        C, L, H, W = video.size()
        rescaled_video = torch.FloatTensor(C, L, h, w)
        
        # use torchvision implemention to resize video frames
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(self.size, self.interpolation),
            torchvision.transforms.ToTensor(),
        ])

        for l in range(L):
            frame = video[:, l, :, :]
            frame = transform(frame)
            rescaled_video[:, l, :, :] = frame

        return rescaled_video

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

        
class VideoRandomCrop(object): # Noe med denne metoden som ikke funker
    """ Crop the given Video Tensor (C x L x H x W) at a random location.

    Args:
        size (sequence): Desired output size like (h, w).
    """

    def __init__(self, size):
        assert len(size) == 2
        self.size = size
    
    def __call__(self, video):
        """ 
        Args:
            video (torch.Tensor): Video (C x L x H x W) to be cropped.

        Returns:
            torch.Tensor: Cropped video (C x L x h x w).
        """

        H, W = video.size()[2:]
        h, w = self.size
        assert H >= h and W >= w  # fikk feil på denne sjekken

        top = np.random.randint(0, H - h)
        left = np.random.randint(0, W - w)

        video = video[:, :, top : top + h, left : left + w]
        
        return video
        
class VideoCenterCrop(object):
    """ Crops the given video tensor (C x L x H x W) at the center.

    Args:
        size (sequence): Desired output size of the crop like (h, w).
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video (C x L x H x W) to be cropped.
        
        Returns:
            torch.Tensor: Cropped Video (C x L x h x w).
        """

        H, W = video.size()[2:]
        h, w = self.size
        assert H >= h and W >= w
        
        top = int((H - h) / 2)
        left = int((W - w) / 2)

        video = video[:, :, top : top + h, left : left + w]
        
        return video
        
class VideoRandomHorizontalFlip(object):
    """ Horizontal flip the given video tensor (C x L x H x W) randomly with a given probability.

    Args:
        p (float): probability of the video being flipped. Default value is 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video to flipped.
        
        Returns:
            torch.Tensor: Randomly flipped video.
        """

        if random.random() < self.p:
            # horizontal flip the video
            video = video.flip([3])

        return video
            

class VideoRandomVerticalFlip(object):
    """ Vertical flip the given video tensor (C x L x H x W) randomly with a given probability.

    Args:
        p (float): probability of the video being flipped. Default value is 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video to flipped.
        
        Returns:
            torch.Tensor: Randomly flipped video.
        """

        if random.random() < self.p:
            # horizontal flip the video
            video = video.flip([2])

        return video
            
class VideoGrayscale(object):
    """ Convert video (C x L x H x W) to grayscale (C' x L x H x W, C' = 1 or 3)

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output video
    """

    def __init__(self, num_output_channels=1):
        assert num_output_channels in (1, 3)
        self.num_output_channels = num_output_channels

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video (3 x L x H x W) to be converted to grayscale.

        Returns:
            torch.Tensor: Grayscaled video (1 x L x H x W  or  3 x L x H x W)
        """

        C, L, H, W = video.size()
        grayscaled_video = torch.FloatTensor(self.num_output_channels, L, H, W)
        
        # use torchvision implemention to convert video frames to gray scale
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Grayscale(self.num_output_channels),
            torchvision.transforms.ToTensor(),
        ])

        for l in range(L):
            frame = video[:, l, :, :]
            frame = transform(frame)
            grayscaled_video[:, l, :, :] = frame

        return grayscaled_video


        