import pandas as pd
import numpy as np
import os

scores_21015 = pd.read_csv('./data/preproc/riqa_scores_21015_norm.csv')
scores_21016 = pd.read_csv('./data/preproc/riqa_scores_21016_norm.csv')

sampled_imgs = os.listdir('./data/interim/')

os.makedirs('./data/interim/good')
os.makedirs('./data/interim/usable')
os.makedirs('./data/interim/reject')

mapping = {0: 'good', 1: 'usable', 2: 'reject'}

for img in sampled_imgs:
    if img.endswith('.png'):
        part_id = img.split('.')[0].split('_')[0]
        field_id = img.split('.')[0].split('_')[1]

        if field_id == '21015':
            scores = scores_21015[(scores_21015['participant_id'] == int(part_id))].values[0]
            
            argmax_idx = np.argmax(scores[1:])
            if scores[argmax_idx+1] > 0.5:
                os.makedirs(f'./data/interim/{mapping[argmax_idx]}/{part_id}')
                os.rename(f'./data/interim/{img}', f'./data/interim/{mapping[argmax_idx]}/{part_id}/{img}')
                np.savetxt(f'./data/interim/{mapping[argmax_idx]}/{part_id}/score.csv', np.array([scores]), delimiter=',')

        if field_id == '21016':
            scores = scores_21016[(scores_21016['participant_id'] == int(part_id))].values[0]
            
            argmax_idx = np.argmax(scores[1:])
            if scores[argmax_idx+1] > 0.5:
                os.makedirs(f'./data/interim/{mapping[argmax_idx]}/{part_id}')
                os.rename(f'./data/interim/{img}', f'./data/interim/{mapping[argmax_idx]}/{part_id}/{img}')
                np.savetxt(f'./data/interim/{mapping[argmax_idx]}/{part_id}/score.csv', np.array([scores]), delimiter=',')





