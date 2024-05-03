from moviepy.editor import *
import json
import os
from pytube import YouTube
from pytube.exceptions import VideoPrivate
from pytube.exceptions import VideoUnavailable
from pytube.cli import on_progress
from pytube import YouTube
from urllib.parse import urlparse, parse_qs
import logging

logging.basicConfig(filename='MSASL/msasl.log',encoding='utf-8',level=logging.INFO, filemode = 'w', format='%(process)d-%(levelname)s-%(message)s') 

logging.info('This will write in basic.log')

logging.info('starting download')
# from http.client import IncompleteRead

# limits number of files to download as not to chew through SSD write cycle
# limit = 99999
limit = 10000
limitedDataset = []
path_to_json = 'MSASL/data/'
train_path = 'MSASL_train.json'

with open(path_to_json+train_path) as f:
    d = json.load(f)
    for i in range(limit):
        limitedDataset.append([d[i]['clean_text'], d[i]['url'], d[i]['start_time'], d[i]['end_time']])
        # logging.info(d[i])

# folder_path = 'C:/your/path/to/folder/'
folder_path = 'MSASL/'
file_extension = 'mp4'
# output paths
dataset_path = 'raw_videos/'
trimmed_dataset_path = 'videos/'

# Get a list of all mp4 files in the folder
files = os.listdir(folder_path + dataset_path)

filenames = []

def loadFilenames():
    # logging.info the list of files
    for file in files:
        filenames.append(os.path.splitext(file)[0])
        # logging.info(filenames)
        # logging.info(len(filenames))
        

fileLookup = {}
toDownload = {}

def query(url):
    try:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        # logging.info(query_params['v'])
        # toDownload.append((query_params['v']))
        return query_params['v']

    except Exception as e:
        logging.info(e)
        url = 'https://' + url
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        return query_params['v']

def getVideoUrl():
    for subarray in limitedDataset:
        queriedLink = query(subarray[1])
        # logging.info('queriedLink ' + queriedLink[0])
        if all([queriedLink[0] not in filenames]):
            toDownload[queriedLink[0]] = []
            # toDownload.append(subarray[1])

def getTimeStamps():
    for subarray in limitedDataset:
        queriedLink = query(subarray[1])
        if all([queriedLink[0] not in filenames]):
            # logging.info(queriedLink[0])
            toDownload.get(queriedLink[0]).append(
                [subarray[0], subarray[2], subarray[3]])
            

# Download videos
def downloadVideo(file):
    i = file
    try:
        path = folder_path + dataset_path
        url = 'https://www.youtube.com/watch?v=' + i
        logging.info(url)
        yt = YouTube(url)
        stream = yt.streams.filter(file_extension='mp4').first()
        

    except VideoUnavailable as e:
        logging.info(e)
        logging.info(i + ' is private')
        
    except Exception as e:
        logging.info(e)
    else:
        if(stream != None):
            # stream.download(filename=i + ".mp4", output_path=path)
            # logging.info('downloading')
            logging.info('is stuff working')
            try:
                logging.info('trying')
                stream.download(filename=i + ".mp4", output_path=path)
                logging.info('downloading')
            except Exception as e:
                logging.info(e)

                
          
getVideoUrl()
getTimeStamps()

for item in toDownload:
    logging.info(item)
    downloadVideo(item)

def getFileLookup():
    return fileLookup