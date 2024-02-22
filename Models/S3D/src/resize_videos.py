import os
import cv2
import glob

from tqdm.auto import tqdm

ROOT_DIR = os.path.join('..', 'input','workout_recognition')
DST_ROOT = os.path.join('..', 'input', 'workout_recognition_resized')
RESIZE_TO = 512
os.makedirs(DST_ROOT, exist_ok=True)

all_videos = glob.glob(os.path.join(ROOT_DIR, '*', '*'), recursive=True)

def resize(image, img_size=512, maintain_aspect_ratio=True):
    if not maintain_aspect_ratio:
        image = cv2.resize(image, (RESIZE_TO, RESIZE_TO))
    else:
        h0, w0 = image.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            image = cv2.resize(image, (int(w0 * r), int(h0 * r)))
    return image

for i, video_path in tqdm(enumerate(all_videos), total=len(all_videos)):
    cap = cv2.VideoCapture(video_path)
    class_name = video_path.split(os.path.sep)[-2]
    video_name = video_path.split(os.path.sep)[-1]
    fps = int(cap.get(5))
    # Get the first frame and calculate resized height and width to create
    # video writer object.
    ret, frame = cap.read()
    frame = resize(frame, img_size=RESIZE_TO, maintain_aspect_ratio=True)
    new_height, new_width, _ = frame.shape
    # Define codec and create VideoWriter object.
    out = cv2.VideoWriter(os.path.join(DST_ROOT, class_name, video_name), 
                        cv2.VideoWriter_fourcc(*'mp4v'), fps, 
                        (new_width, new_height))

    if not os.path.exists(os.path.join(DST_ROOT, class_name)):
        os.makedirs(os.path.join(DST_ROOT, class_name))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = resize(frame, img_size=RESIZE_TO, maintain_aspect_ratio=True)
            out.write(frame)
        else:
            break