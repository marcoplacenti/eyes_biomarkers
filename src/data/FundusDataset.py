# -*- coding: utf-8 -*-
import logging
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
from PIL import Image
import os
import numpy as np
import joblib
import json
import random

class FundusDataset(Dataset):
    def __init__(self, set='train', labels=False, size='full', 
                normalized=False, resolution=224, augmented=True):
        self.set = set
        self.labels = labels
        self.size = size
        self.norm = normalized
        self.res = resolution
        self.aug = augmented

        if self.norm:
            self.transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225]
                                )
                            ])
            
        else:
            self.transform = transforms.Compose([
                                transforms.ToTensor()
                            ])
        
        self.data_dir = f'/nfs_home/projects/shared_projects/eye_imaging/data/'
        if self.res == 299:
            if self.aug:
                self.data_dir += '2022_eyes_biomarkers/processed_aug_299x299_final'
            else:
                self.data_dir += '2022_eyes_biomarkers/processed_299x299_final'
        elif self.res == 224:
            if self.aug:
                self.data_dir += '2022_eyes_biomarkers/processed_aug_224x224_final'
            else:
                self.data_dir += '2022_eyes_biomarkers/processed_224x224_final'
                #self.data_dir += '2022_eyes_biomarkers/processed_aug_noflip'
        
        self.images_path = self.data_dir#+'/'+self.set

        self.images = os.listdir(self.images_path)
        
        # using only participants with both eyes scans
        from collections import Counter
        counter = Counter([str(img.split('_')[0]) for img in self.images])
        self.valid_part_idx = []
        for k in list(counter.keys()):
            if counter[str(k)] == (10 if self.aug else 2):
                self.valid_part_idx.append(str(k))

        if self.size != 'full':
            self.valid_part_idx = random.sample(self.valid_part_idx, self.size)

        self.images = [img for img in self.images 
                            if str(img.split('_')[0]) in self.valid_part_idx]

        if self.set == 'inference':
            if self.aug:
                self.images = [im for im in self.images if im.endswith('_0')]

        self.valid_part_idx = np.unique(
                            [str(img.split('_')[0]) for img in self.images]
        )

        self.gen_mapping()

        #self.rgb_dir = f'/nfs_home/projects/shared_projects/eye_imaging/data/2022_oct_fundus/processed/fundus'

    def gen_mapping(self):
        self.item_to_class_map = {}
        for item_idx, item in enumerate(self.valid_part_idx):
            self.item_to_class_map[str(int(item))] = item_idx

        self.class_to_items_map = {}
        self.img_to_idx_map = {}
        for item_idx, item in enumerate(self.images):
            self.img_to_idx_map[item] = item_idx
            item_class = self.item_to_class_map[item.split('_')[0]]
            if item_class not in list(self.class_to_items_map.keys()):
                self.class_to_items_map[item_class] = [item]
            else:
                self.class_to_items_map[item_class].append(item)
            
                
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.fetchitem(idx)
        elif isinstance(idx, tuple):
            return (self.fetchitem(idx[0]), self.fetchitem(idx[1]))
            
    def fetchitem(self, idx):
        img_filename = self.images[idx]  
        img = joblib.load('/'.join([self.images_path, img_filename]))#/255
        #img = img.crop((0, 35, 0+128, 35+128))
        #img.save(f'./data/test_{img_filename}.png')
        img = self.transform(img).float()

        part_id = str(int(img_filename.split('_')[0]))
        target = self.item_to_class_map[part_id]

        return idx, img, target#, part_id, img_filename.split('_')[1]

    def classes_count(self):
        return len(list(self.item_to_class_map.values()))

    def get_sorted_dataset(self):
        return self.images
    
    def get_item_to_class_map(self):
        return self.item_to_class_map

    def get_class_to_item_map(self):
        return self.class_to_items_map

    def get_img_to_idx_map(self):
        return self.img_to_idx_map

    def get_part_from_idx(self, idx):
        return self.images[idx]
    