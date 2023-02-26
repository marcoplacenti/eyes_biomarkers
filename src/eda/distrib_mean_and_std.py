import torch
import joblib
import numpy as np
import os
import csv
import torchvision.transforms as transforms

data_dir = '/nfs_home/projects/shared_projects/eye_imaging/data/2022_eyes_biomarkers/processed/train/'
images = os.listdir(data_dir)

transform = transforms.Compose([transforms.ToTensor()])

means, stds = torch.zeros(3), torch.zeros(3)
for idx, image in enumerate(images):
    with open('prog.txt', 'a+') as prog:
        writer = csv.writer(prog)
        writer.writerows(str(idx)+'\n')
    data = joblib.load(data_dir+image)/255
    data = np.transpose(data, (2,1,0))
    data = transform(data)
    mean, std = data.mean([1,2]), data.std([1,2])
    means += mean
    stds += std

print('Mean: ', means/len(images))
print('STD: ', stds/len(images))