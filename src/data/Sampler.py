import torch
from torch.utils.data.sampler import Sampler
import random
import copy

class TripletSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

        self.images = self.data_source.images
        self.item_to_class = self.data_source.get_item_to_class_map()
        self.class_to_items = self.data_source.get_class_to_item_map()
        self.img_to_idx = self.data_source.get_img_to_idx_map()
        self.my_list = []
        for img in self.images:
            pos_samples_idx = self.find_positive_samples(img)
            self.my_list.extend(pos_samples_idx)
            neg_samples_idx = self.find_negative_samples(img, len(pos_samples_idx))
            self.my_list.extend(neg_samples_idx)

    def __iter__(self):
        return iter(self.my_list)

    def __len__(self):
        return len(self.data_source)

    def find_positive_samples(self, img):
        anchor_class = self.item_to_class[img.split('_')[0]]
        pos_samples = copy.copy(self.class_to_items[anchor_class])
        pos_samples.remove(img)
        pos_samples = random.sample(pos_samples, 1)
        pos_samples.append(img)
        pos_samples_idx = [self.img_to_idx[sample] for sample in pos_samples]
        return pos_samples_idx

    def find_negative_samples(self, img, pos_length):
        anchor_class = self.item_to_class[img.split('_')[0]]
        neg_candidates = copy.copy(self.class_to_items)
        neg_candidates = list(neg_candidates.keys())
        neg_candidates.remove(anchor_class)
        neg_samples_class = random.sample(neg_candidates, int((self.batch_size-pos_length)/2))
        neg_samples = [random.sample(self.class_to_items[neg_class], 2) 
                            for neg_class in neg_samples_class]
        neg_samples = [item for sublist in neg_samples for item in sublist]
        neg_samples_idx = [self.img_to_idx[sample] for sample in neg_samples]
        return neg_samples_idx


class MocoSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

        self.images = self.data_source.images
        self.item_to_class = self.data_source.get_item_to_class_map()
        self.class_to_items = self.data_source.get_class_to_item_map()
        self.img_to_idx = self.data_source.get_img_to_idx_map()
        self.encoding_indexes = []
        self.momentum_indexes = []
        for img in self.images:
            self.encoding_indexes.append(self.img_to_idx[img])
            q_class = self.item_to_class[img.split('_')[0]]
            candidates_samples = copy.copy(self.class_to_items[q_class])
            candidates_samples.remove(img)
            chosen_candidate = random.sample(candidates_samples, 1)[0]
            self.momentum_indexes.append(self.img_to_idx[chosen_candidate])

        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.encoding_indexes):
            self.index = 0
            raise StopIteration
        result = (self.encoding_indexes[self.index], self.momentum_indexes[self.index])
        self.index += 1
        return result

    def __len__(self):
        return len(self.encoding_indexes)
