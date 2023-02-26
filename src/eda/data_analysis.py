import os
import numpy as np

data_dir = '/nfs_home/projects/shared_projects/eye_imaging/data/image_data/eye_OCT/'
left_eye_dir = data_dir + '21015'
right_eye_dir = data_dir + '21016'

left = os.listdir(left_eye_dir)
left_dict = {'participants': [], 'instances': [], 'array': []}
for entry in left:
    entry = entry.split('_')
    if entry[-1].endswith('.png'):
        left_dict['participants'].extend([entry[0]])
        left_dict['instances'].extend([entry[2]])
        left_dict['array'].extend([entry[3].split('.')[0]])
    
print('Number of participants left eye: ', np.unique(left_dict['participants']).shape)

right = os.listdir(right_eye_dir)
right_dict = {'participants': [], 'instances': [], 'array': []}
for entry in right:
    entry = entry.split('_')
    if entry[-1].endswith('.png'):
        right_dict['participants'].extend([entry[0]])
        right_dict['instances'].extend([entry[2]])
        right_dict['array'].extend([entry[3].split('.')[0]])
    
print('Number of participants right eye: ', np.unique(right_dict['participants']).shape)

print('Number of participants in left but not right: ' , len(list(set(left_dict['participants']).difference(right_dict['participants']))))
print('Number of participants in right but not left: ', len(list(set(right_dict['participants']).difference(left_dict['participants']))))

both_eyes_parts = list(set([item for item in left_dict['participants'] if item in right_dict['participants']]))
print('Number of participants having both eyes scanned: ' , len(both_eyes_parts))


import pandas as pd
train_set = pd.read_csv('./data/meta/train_set.csv')
test_set = pd.read_csv('./data/meta/test_set.csv')
val_set = pd.read_csv('./data/meta/val_set.csv')
full_set = pd.concat([train_set, test_set, val_set])
print(full_set)

genders = {'1': 0, '0': 0}
ht = {'1': 0, '0': 0}
cross_classes = {'00': 0, '01': 0, '10': 0, '11': 0}
for part in both_eyes_parts:
    record = full_set.loc[full_set['participant_id'] == int(part)].head(1)
    gender = record['gender'].values
    sbp = record['sbp'].values
    try:
        gender = gender[0]
        sbp = sbp[0]
    except:
        continue
    if gender == 1 and sbp >= 140:
        genders['1'] += 1
        ht['1'] += 1
        cross_classes['11'] += 1
    if gender == 1 and sbp < 140:
        genders['1'] += 1
        ht['0'] += 1
        cross_classes['10'] += 1
    if gender == 0 and sbp >= 140:
        genders['0'] += 1
        ht['1'] += 1
        cross_classes['01'] += 1
    if gender == 0 and sbp < 140:
        genders['0'] += 1
        ht['0'] += 1
        cross_classes['00'] += 1
    
print(genders)
print(ht)
print(cross_classes)