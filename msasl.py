from moviepy.editor import *
import json
import os
from pytube import YouTube
from pytube.exceptions import VideoPrivate
from pytube.exceptions import VideoUnavailable
from pytube.cli import on_progress
from pytube import YouTube
from urllib.parse import urlparse, parse_qs

# limits number of files to download as not to chew through SSD write cycle
# limit = 99999
limit = 20
limitedDataset = []
path_to_json = '/Users/ingridmariewolneberg/Desktop/msasl_jsons/'
train_path = 'MSASL_train.json'

with open(path_to_json+train_path) as f:
    d = json.load(f)
    for i in range(limit):
        limitedDataset.append([d[i]['clean_text'], d[i]['url'], d[i]['start_time'], d[i]['end_time']])
        # print(d[i])

# folder_path = 'C:/your/path/to/folder/'
folder_path = '/Users/ingridmariewolneberg/Desktop/msasl/'
file_extension = 'mp4'

# output paths
dataset_path = 'dataset/'
trimmed_dataset_path = 'trimmed_dataset/'

# Get a list of all mp4 files in the folder
files = os.listdir(folder_path + dataset_path)

filenames = []

def loadFilenames():
    # Print the list of files
    for file in files:
        filenames.append(os.path.splitext(file)[0])
        # print(filenames)
        # print(len(filenames))
        

fileLookup = {}
toDownload = {}

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

def getVideoUrl():
    for subarray in limitedDataset:
        queriedLink = query(subarray[1])
        # print('queriedLink ' + queriedLink[0])
        if all([queriedLink[0] not in filenames]):
            toDownload[queriedLink[0]] = []
            # toDownload.append(subarray[1])

def getTimeStamps():
    for subarray in limitedDataset:
        queriedLink = query(subarray[1])
        if all([queriedLink[0] not in filenames]):
            # print(queriedLink[0])
            toDownload.get(queriedLink[0]).append(
                [subarray[0], subarray[2], subarray[3]])
            

# Download videos
def downloadVideo(file):
    i = file
    try:
        path = folder_path + dataset_path
        url = 'https://www.youtube.com/watch?v=' + i
        print(url)
        yt = YouTube(url)
        stream = yt.streams.filter(file_extension='mp4').first()
        

    except VideoUnavailable as e:
        print(e)
        print(i + ' is private')
        
    except Exception as e:
        print(e)

    else:
        if(stream != None):
            stream.download(filename=i + ".mp4", output_path=path)
            print('downloading')

getVideoUrl()
getTimeStamps()

for item in toDownload:
    downloadVideo(item)

def getFileLookup():
    return fileLookup