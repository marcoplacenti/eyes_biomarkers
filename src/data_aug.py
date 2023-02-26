import torchvision.transforms as transforms
import torch
import numpy as np
import os
import joblib
import shutil

data_dir = '/nfs_home/projects/shared_projects/eye_imaging/data/'

src_dir = data_dir + '2022_eyes_biomarkers/processed_299x299_final/'

transform = transforms.Compose([
                        transforms.RandomApply([
                            transforms.RandomRotation(15)
                        ], p=1),
                        transforms.RandomApply([
                            transforms.ColorJitter(0.1, 0.3, 0.1, 0)
                        ], p=1),
                        #transforms.ToTensor()
                        
])

target_dir = data_dir + '2022_eyes_biomarkers/processed_aug_299x299_final/'
old_aug_dir = os.listdir(data_dir + '2022_eyes_biomarkers/processed_aug_299x299/')
part_old_aug_dir = np.unique([file.split('_')[0] for file in old_aug_dir])
for idx, file in enumerate(os.listdir(src_dir)):
    if file in part_old_aug_dir:
        for i in range(5):
            shutil.copyfile(old_aug_dir+'/'+file+'_'+str(i), target_dir+'/'+file+'_'+str(i))        
    else:
        shutil.copyfile(src_dir+'/'+file, target_dir+'/'+file+'_0')
        img = joblib.load(src_dir+'/'+file)

        for i in range(4):
            aug_img = transform(img)
            joblib.dump(aug_img,
                f'{target_dir}/{file}_{i+1}',
                compress=5)
