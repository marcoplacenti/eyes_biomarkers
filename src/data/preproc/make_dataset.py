# -*- coding: utf-8 -*-
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np

class Dataset(Dataset):
    def __init__(self, field_id, dim=(224, 224)):
        self.field_id = field_id
        self.dim = dim
        
        self.color = cv2.IMREAD_COLOR

        self.scan_info = pd.read_csv('./data/meta/post_qc_scan_instance_set.csv')
        
        if self.field_id == '21015':
            self.col = 'left_eye_crop'
        elif self.field_id == '21016':
            self.col = 'right_eye_crop'

        self.scan_info = self.scan_info[self.scan_info[self.col] != '[]']

        self.img_dir = '/nfs_home/projects/shared_projects/eye_imaging/data/image_data/eye_OCT/'
        self.img_dir += self.field_id
        
    def __len__(self):
        return len(self.scan_info)

    def __getitem__(self, idx):
        item_info = self.scan_info.iloc[idx]
        filename = '_'.join([str(item_info['participant_id']), self.field_id, str(item_info['instance']), '0'])
        img_path = os.path.join(self.img_dir, filename)
        src = cv2.imread(img_path+'.png', self.color)

        coord = self.scan_info.iloc[idx][self.col].replace('[','').replace(']','').split(',')
        coord = [int(v) for v in coord]     

        cropped_img = src[coord[0]:coord[0]+coord[2], coord[1]:coord[1]+coord[3], :]

        transform_image = cv2.resize(cropped_img, (224, 224))
        
        return torch.from_numpy(transform_image).permute(2, 1, 0).float() # permutation from bgr to rgb

    def get_data(self):
        return self.scan_info

    def get_items(self, idxs, batch_size):
        return self.scan_info.iloc[idxs]['participant_id'].values.reshape((batch_size,1))
