import os
import json
from moviepy.editor import *
from urllib.parse import urlparse, parse_qs
import csv

limit = 10000
limitedDataset = []
path_to_json = 'MSASL/data/'
train_path = 'MSASL_train.json'
folder_path = 'MSASL/'
file_extension = 'mp4'
# output paths
dataset_path = 'raw_videos/'
trimmed_dataset_path = 'videos/'

with open(path_to_json+train_path) as f:
    d = json.load(f)
    for i in range(limit):
        limitedDataset.append([d[i]['clean_text'], d[i]['url'], d[i]['start_time'], d[i]['end_time']])
        # print(d[i])

# Trim videos
# Get a list of all mp4 files in the folder
files = os.listdir(folder_path + dataset_path)
videonames = []
availableVideos = []

# Get a list of all mp4 files in trimmed_dataset
trimmedFiles = os.listdir(folder_path + trimmed_dataset_path)
trimmedVideoNames = []
file_extension = 'mp4'

fileLookup = {}

filenames = []

def query(url):
    try:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        # print(query_params['v'])
        # toDownload.append((query_params['v']))
        return query_params['v']

    except Exception as e:
        print(e)
        url = 'https://' + url
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        return query_params['v']

def loadFilenames():
    # Print the list of files
    for file in files:
        filenames.append(os.path.splitext(file)[0])
        # print(filenames)
        # print(len(filenames))

def getAllVideoUrl():
    for subarray in limitedDataset:
        queriedLink = query(subarray[1])
        # print('queriedLink ' + queriedLink[0])
        if all([queriedLink[0] in filenames]):
            fileLookup[queriedLink[0]] = []
            # print(toDownload)
            # toDownload.append(subarray[1])

def getAllTimeStamps():
    for subarray in limitedDataset:
        queriedLink = query(subarray[1])
        if all([queriedLink[0] in filenames]):
            fileLookup.get(queriedLink[0]).append(
                [subarray[0], subarray[2], subarray[3]])

def getDatasetNames():
# Print the list of dataset files
    for file in files:
        videonames.append(os.path.splitext(file)[0])

def getTrimmedDatasetNames():
    # Print the list of files
    for file in trimmedFiles:
        trimmedVideoNames.append(os.path.splitext(file)[0])

def trimVid(name, segments, id, writer):
    print(name)
    print(segments)
    # # outputName = segments[0][0]
    # startTime = segments[0][1]
    # endTime = segments[0][2]
    # try: 
    #     video = VideoFileClip(folder_path + dataset_path + name +".mp4").subclip(startTime, endTime)
    #     video.write_videofile(f'{folder_path}{trimmed_dataset_path}{id}.mp4', fps=25)
    # except Exception as e:
    #     print(e)

    for i in segments:
        print(i)
        outputName = i[0]
        startTime = i[1]
        endTime = i[2]
        try: 
            if f'{id}' not in trimmedVideoNames:
                video = VideoFileClip(folder_path + dataset_path + name +".mp4").subclip(startTime, endTime)
                video.write_videofile(f'{folder_path}{trimmed_dataset_path}{id}.mp4', fps=25)
                writer.writerow([id, outputName])
                id+=1
            else: 
                print('already exists')
        except Exception as e:
            print(e)
        # # check if file exists in trimmed-dataset
        # if any([outputName in trimmedVideoNames]):
        #     print('file exists')
        #     print('I should be merging ' + outputName)
        #     video_2 = VideoFileClip(folder_path + trimmed_dataset_path + outputName +".mp4")
        #     finalvideo = concatenate_videoclips([video, video_2])
        #     finalvideo.write_videofile(folder_path + trimmed_dataset_path + outputName + ".mp4", fps=25)
        # else:
        #     video.write_videofile(folder_path + trimmed_dataset_path + outputName + ".mp4", fps=25)
    return id

loadFilenames()

print()
getAllVideoUrl()
getAllTimeStamps()


getDatasetNames()
getTrimmedDatasetNames()

# print(fileLookup)
id = 0
with open('MSASL/x.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["video_id", "label"]
    writer.writerow(field)
    for item in fileLookup.keys():
        segments = fileLookup[item]
        new_id = trimVid(item, segments, id, writer)
        id = new_id
        print(id)
