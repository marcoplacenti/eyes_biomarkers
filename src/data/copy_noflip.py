import os
import shutil
import joblib
from PIL import Image

proc_aug_224 = '/nfs_home/projects/shared_projects/eye_imaging/data/2022_eyes_biomarkers/processed_aug_299x299'
dest_augnoflip_224 = '/nfs_home/projects/shared_projects/eye_imaging/data/2022_eyes_biomarkers/processed_aug_noflip_299x299'

for file in os.listdir(proc_aug_224):
    if file.split('_')[1] == '21016':
        img = joblib.load(proc_aug_224+'/'+file)
        transform_image = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        joblib.dump(
                transform_image, 
                dest_augnoflip_224+'/'+file,
                compress=5)
    else:
        shutil.copy(proc_aug_224+'/'+file, dest_augnoflip_224+'/'+file)
        