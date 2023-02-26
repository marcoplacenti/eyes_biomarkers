import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import cv2
import os
import multiprocessing

class InstanceScanFilter():
    def __init__(self, data_path, separator):
        self.instances_info = pd.read_csv(data_path, sep=separator)

    def filter(self):
        instances_info = self.instances_info.copy()
        # dropping date of third and fourth scan
        instances_info = instances_info.drop(columns=[
        'f.53.2.0','f.53.3.0'])#,'f.21015.0.1','f.21015.1.1',
                        #'f.21016.0.1','f.21016.1.1','f.21017.0.1','f.21017.1.1',
                        #'f.21018.0.1','f.21018.1.1'])

        # dropping patients not having any fundus scanned either in first measurement or follow up
        instances_info = instances_info.drop(instances_info[
                                (instances_info['f.21015.0.0'].isnull()) & 
                                (instances_info['f.21015.1.0'].isnull()) & 
                                (instances_info['f.21016.0.0'].isnull()) & 
                                (instances_info['f.21016.1.0'].isnull())].index)

        # keeping patients having both fundus and oct in at least one eye (either in first measurement or in follow up) 
        instances_info = instances_info.loc[
                        ((~instances_info['f.21015.0.0'].isnull())&(~instances_info['f.21017.0.0'].isnull())) | 
                        ((~instances_info['f.21016.0.0'].isnull())&(~instances_info['f.21018.0.0'].isnull())) |
                        ((~instances_info['f.21015.1.0'].isnull())&(~instances_info['f.21017.1.0'].isnull())) | 
                        ((~instances_info['f.21016.1.0'].isnull())&(~instances_info['f.21018.1.0'].isnull()))]

        #print(instances_info)
        #print(instances_info[~instances_info['f.21015.0.1'].isnull() & instances_info['f.21015.0.0'].isnull()]) # should be empty

        # mask for patients having fundus first measurement in at least one eye
        mask_instance00 = (~instances_info['f.21015.0.0'].isnull()) | (~instances_info['f.21016.0.0'].isnull())
        # mask for patients having fundus first measurement taken again in at least one eye
        mask_instance01 = (~instances_info['f.21015.0.1'].isnull()) | (~instances_info['f.21016.0.1'].isnull())

        # mask for patients having fundus follow up (IMPORTANT: one patient may have both) in at least one eye
        mask_instance10 = (~instances_info['f.21015.1.0'].isnull()) | (~instances_info['f.21016.1.0'].isnull())
        # mask for patients having fundus follow up taken again in at least one eye
        mask_instance11 = (~instances_info['f.21015.1.1'].isnull()) | (~instances_info['f.21016.1.1'].isnull())

        # mask for patients having left eye's both fundus and oct in first measurements
        mask_left_instance00 = mask_instance00 & (~instances_info['f.21015.0.0'].isnull()) & (~instances_info['f.21017.0.0'].isnull())
        # mask for patients having right eye's both fundus and oct in first measurements
        mask_right_instance00 = mask_instance00 & (~instances_info['f.21016.0.0'].isnull()) & (~instances_info['f.21018.0.0'].isnull())
        # mask for patients having left eye's both fundus and oct in follow up
        mask_left_instance10 = mask_instance10 & (~instances_info['f.21015.1.0'].isnull()) & (~instances_info['f.21017.1.0'].isnull())
        # mask for patients having right eye fundus and oct in follow up
        mask_right_instance10 = mask_instance10 & (~instances_info['f.21016.1.0'].isnull()) & (~instances_info['f.21018.1.0'].isnull()) 

        # mask for patients having both eyes fundus and oct in first measurement
        use_instance_00 = mask_left_instance00 & mask_right_instance00
        # mask for patients having both eyes fundus and oct in follow up
        use_instance_10 = mask_left_instance10 & mask_right_instance10

        # mask for patients NOT having both eyes fundus and oct in first measurement
        no_instance_00 = (~mask_left_instance00 & ~mask_right_instance00)

        # adding above info in dataframe
        instances_info['instance'] = 0
        instances_info['left_eye'] = 0
        instances_info['right_eye'] = 0
        instances_info.loc[use_instance_10, 'instance'] = 1
        instances_info.loc[use_instance_00, 'instance'] = 0
        instances_info.loc[no_instance_00, 'instance'] = 1
        instances_info.loc[(instances_info['instance'] == 1), 'scan_date'] = instances_info['f.53.1.0']
        instances_info.loc[(instances_info['instance'] == 0), 'scan_date'] = instances_info['f.53.0.0']

        instances_info.loc[use_instance_10, 'left_eye'] = 1
        instances_info.loc[use_instance_00, 'left_eye'] = 1
        instances_info.loc[use_instance_10, 'right_eye'] = 1
        instances_info.loc[use_instance_00, 'right_eye'] = 1
        instances_info.loc[mask_left_instance10 & (instances_info['instance']==1), 'left_eye'] = 1
        instances_info.loc[mask_left_instance00 & (instances_info['instance']==0), 'left_eye'] = 1
        instances_info.loc[mask_right_instance10 & (instances_info['instance']==1), 'right_eye'] = 1
        instances_info.loc[mask_right_instance00 & (instances_info['instance']==0), 'right_eye'] = 1

        #print(instances_info)

        instances_info['array_21015'] = 0
        instances_info['array_21016'] = 0
        instances_info['array_21017'] = 0
        instances_info['array_21018'] = 0

        mask_array_21015 = ((~instances_info['f.21015.0.1'].isnull() & ~instances_info['instance']) | 
                            (~instances_info['f.21015.1.1'].isnull() & instances_info['instance']))
        mask_array_21016 = ((~instances_info['f.21016.0.1'].isnull() & ~instances_info['instance']) |
                            (~instances_info['f.21016.1.1'].isnull() & instances_info['instance']))
        mask_array_21017 = ((~instances_info['f.21017.0.1'].isnull() & ~instances_info['instance']) |
                            (~instances_info['f.21017.1.1'].isnull() & instances_info['instance']))
        mask_array_21018 = ((~instances_info['f.21018.0.1'].isnull() & ~instances_info['instance']) |
                            (~instances_info['f.21018.1.1'].isnull() & instances_info['instance'])) 

        instances_info.loc[mask_array_21015, 'array_21015'] = 1
        instances_info.loc[mask_array_21016, 'array_21016'] = 1
        instances_info.loc[mask_array_21017, 'array_21017'] = 1
        instances_info.loc[mask_array_21018, 'array_21018'] = 1

        instances_info = instances_info.reset_index().drop(columns='index')

        return instances_info


def proc_img(field_id, img, idx):
    print(idx)
    img_dir = '/nfs_home/projects/shared_projects/eye_imaging/data/image_data/eye_OCT/'+field_id
    entry_info = []

    file_path = os.path.join(img_dir, img)+'.png'
    entry_info.append(img)
    src = cv2.imread(file_path, 1)#, cv2.IMREAD_COLOR)
    #cv2.imwrite(f'./data/dummy_test_{idx}_orig.png', src)

    item = cv2.medianBlur(src, 5) # to reduce noise in order to avoid false circles detection
    item = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite(f'./data/dummy_test_{idx}_blur&grey.png', item)
    circles = cv2.HoughCircles(item, cv2.HOUGH_GRADIENT,1.5,300,
                            param1=50, # it is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller)
                            param2=30, # it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
                            minRadius=400,
                            maxRadius=800)

    #print(img)
    #cv2.imwrite(f'./data/interim/{img}.png', item)#.detach().numpy()[0,])
    if circles is None:
        entry_info.append('0') # if no circle detected, use a 0 flag
        entry_info.extend([str(np.NAN), str(np.NAN), str(np.NAN), str(np.NAN)]) # empty coordinates
        return entry_info, None

    entry_info.append('1') # if cirlce is detected, use a 1 flag
    
    circles = np.uint16(np.around(circles))
    r = circles[0,0,2]
    x = circles[0,0,1] - r
    y = circles[0,0,0] - r

    """
    src_copy = src.copy()
    for i in circles[0, :]:
        center = (i[0], i[1])
        cv2.circle(src_copy, center, 1, (0, 100, 100), 3)
        radius = i[2]
        cv2.circle(src_copy, center, radius, (255, 0, 255), 3)
        #cv2.imwrite(f'./data/dummy_test_{idx}_src_w_circles.png', src_copy)
        break
    """

    cropped_img = src[x:(x+2*r), y:(y+2*r), :]
    #cv2.imwrite(f'./data/dummy_test_{idx}_cropped.png', cropped_img)
    entry_info.extend([str(x), str(y), str(x+2*r), str(y+2*r)])

    transformed_image = cv2.resize(cropped_img, (224, 224))
    #cv2.imwrite(f'./data/raw/{img}.png', transformed_image)#.detach().numpy()[0,])

    #print(entry_info)
    #print(len(entry_info))
    return entry_info, transformed_image

if __name__ == '__main__':

    data_path = '/nfs_home/projects/shared_projects/eye_imaging/code/scan_instance_raw'
    instance_scan_filter = InstanceScanFilter(data_path, '\t')
    instances_info = instance_scan_filter.filter()
    instances_info.to_csv('./data/meta/candidate_patients_info.csv', index=False)

    for field_id in ['21015', '21016']:#, '21017', '21018']:
             
        cols = list(instances_info.columns)[:3]
        cols.extend([col for col in instances_info.columns if col.startswith(f'f.{field_id}') or col=='instance'])
        cols.append(f'array_{field_id}')
        instances_info_reduced = instances_info[cols]
        valid_instances = instances_info_reduced[
                                            (~instances_info_reduced[f'f.{field_id}.0.0'].isnull() | 
                                            ~instances_info_reduced[f'f.{field_id}.0.1'].isnull() |
                                            ~instances_info_reduced[f'f.{field_id}.1.0'].isnull() |
                                            ~instances_info_reduced[f'f.{field_id}.1.1'].isnull())]

        imgs = ['_'.join([line[0], field_id, line[-2], line[-1]]) for line in valid_instances.values.astype(str)]

        pool_obj = multiprocessing.Pool()
        args = [(field_id, img, idx) for idx, img in enumerate(imgs)]
        result = pool_obj.starmap(proc_img, args)

        #print(result)
        circles_detector_summary, proc_images = [], []
        for res in result:
            circles_detector_summary.append(np.array(res[0]))
            if res[1] is not None:
                proc_images.append([res[0][0], res[1]])

        circles_detector_summary = pd.DataFrame(
                                    data=circles_detector_summary,
                                    columns=[
                                        'image_path', 'circle_detected_flag',
                                        'x', 'y', 'w', 'h'
                                    ]
        )
        circles_detector_summary.to_csv(f'./data/meta/circles_detector_sum_{field_id}.csv')

        np.savez_compressed(f'./data/processed/proc_images_{field_id}.npz', a=np.array(proc_images))
