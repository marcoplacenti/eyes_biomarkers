import pandas as pd
import numpy as np
import json
import os
import cv2
import joblib
from PIL import Image

dict = {}

def process_data_split(dataset):

    part_identifiers = dataset['participant_id'].values
    
    color = cv2.IMREAD_COLOR
    data_dir = '/nfs_home/projects/shared_projects/eye_imaging/data/'
    img_dir = data_dir + 'image_data/eye_OCT/'
    dest_dir_299 = data_dir + '2022_eyes_biomarkers/processed_299x299_final'
    dest_dir_224 = data_dir + '2022_eyes_biomarkers/processed_224x224_final'
    dest_dir_224_noflip = data_dir + '2022_eyes_biomarkers/processed_224x224_noflip_final'

    riqa_left = pd.read_csv(f'./data/preproc/riqa_scores_21015_norm.csv')
    riqa_right = pd.read_csv(f'./data/preproc/riqa_scores_21016_norm.csv')
    
    riqa_left_valid = riqa_left[(2*riqa_left['reject'] < riqa_left['usable']+riqa_left['good'])]
    riqa_right_valid = riqa_right[(2*riqa_right['reject'] < riqa_right['usable']+riqa_right['good'])] 
    
    riqa_left_valid_participants = riqa_left_valid['participant_id'].values
    riqa_right_valid_participants = riqa_right_valid['participant_id'].values
    
    final_valid_participants = [part for part in riqa_left_valid_participants 
                                    if part in riqa_right_valid_participants]
    
    for idx, part in enumerate(final_valid_participants):
        with open(f'./progress.log', 'w') as of:
            of.write(f'Processing: {idx+1}/{len(final_valid_participants)}\n')

        try:
            part_data = dataset.loc[dataset['participant_id'] == part].head(1).values[0]
            instance = str(part_data[1])

            for field_id in ['21015', '21016']:
                filename = '_'.join([str(int(part)), field_id, instance, '0'])
                img_path = os.path.join(img_dir+str(field_id), filename)
                src = Image.open(img_path+'.png').convert('RGB')

                coord_col = (4 if field_id == '21015' else 5)
                coord = part_data[coord_col].replace('[','').replace(']','').split(',')
                coord = [int(v) for v in coord]     

                cropped_img = src.crop((coord[1], coord[0], coord[2]+coord[1], coord[3]+coord[0]))

                transform_image_299x299 = cropped_img.resize((299,299))
                transform_image_224x224 = cropped_img.resize((224,224))
                transform_image_224x224_noflip = cropped_img.resize((224,224))

                if field_id == '21016':
                    transform_image_224x224 = transform_image_224x224.transpose(Image.FLIP_LEFT_RIGHT)
                    transform_image_299x299 = transform_image_299x299.transpose(Image.FLIP_LEFT_RIGHT)

                if not os.path.exists(f'{dest_dir_224}'):
                    os.makedirs(f'{dest_dir_224}')
                joblib.dump(
                        transform_image_224x224, 
                        f'{dest_dir_224}/{str(int(part))}_{field_id}',
                        compress=5
                )

                if not os.path.exists(f'{dest_dir_299}'):
                    os.makedirs(f'{dest_dir_299}')
                joblib.dump(
                        transform_image_299x299, 
                        f'{dest_dir_299}/{str(int(part))}_{field_id}',
                        compress=5
                )

                if not os.path.exists(f'{dest_dir_224_noflip}'):
                    os.makedirs(f'{dest_dir_224_noflip}')
                joblib.dump(
                        transform_image_224x224_noflip, 
                        f'{dest_dir_224_noflip}/{str(int(part))}_{field_id}',
                        compress=5
                )
        except IndexError:
            print(f"Error for participant {part}. Skipped.")


def run():
    val = pd.read_csv(f'./data/meta/val_set.csv')
    test = pd.read_csv(f'./data/meta/test_set.csv')
    train = pd.read_csv(f'./data/meta/train_set.csv')
    full_set = pd.concat([train, test, val])

    process_data_split(full_set)

if __name__ == '__main__':
    run()