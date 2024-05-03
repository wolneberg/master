from MSASL.tf_preprocess import get_frame_set, format_dataset

num_frames = 20
resolution = 172
frame_step = 2
csv_file = 'MSASL/msasl.csv'

train_set = get_frame_set(num_frames, resolution, frame_step, csv_file)

train_dataset = format_dataset(train_set)

print(train_set)

print(train_dataset[0])
print(train_dataset[56])