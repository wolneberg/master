import pandas as pd

# Preprocessing of WLASL
def get_video_subset(wlasl_samples, subset):
    videos = pd.read_json(f'WLASL/data/{wlasl_samples}.json').transpose()
    train_videos = videos[videos['subset'].str.contains(subset)].index.values.tolist()
    return train_videos

def get_missing_videos():
    f = open('WLASL/data/missing.txt', 'r')
    missing = []
    for line in f:
        missing.append(line.strip())
    f.close()
    return missing

def get_glosses():
    glosses = pd.read_json('WLASL/data/WLASL_v0.3.json')
    glosses = glosses.explode('instances').reset_index(drop=True).pipe(
        lambda x: pd.concat([x, pd.json_normalize(x['instances'])], axis=1)).drop(
        'instances', axis=1)[['gloss', 'video_id']]
    f = open('WLASL/data/wlasl_class_list.txt', 'r')
    gloss_set = []
    for line in f:
        new_line = line.strip().split('\t')
        new_line[0] = int(new_line[0])
        gloss_set.append(new_line)
    f.close()
    gloss_set = pd.DataFrame(gloss_set, columns=['gloss_id', 'gloss'])
    glosses = gloss_set.merge(glosses, on='gloss')
    #glosses = glosses.drop('gloss', axis=1)
    return glosses