import torch
from models.preproc.DenseNetMCS import dense121_mcs
from data.preproc.make_dataset import Dataset

import csv
import numpy as np
import pandas as pd
import cv2

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = dense121_mcs(n_class=3)

    loaded_model = torch.load(
        '/nfs_home/projects/shared_projects/eye_imaging/code/' +
        '2022_eyes_biomarkers/models/DenseNet121_v3_v1.tar', 
        map_location=torch.device(device)
    )

    model.load_state_dict(loaded_model['state_dict'])
    model.to(device)
    return model

def get_dataloader(field_id):
    dataset = Dataset(field_id)
    batch_size = 8
    loader = torch.utils.data.DataLoader(
                    dataset=dataset, 
                    batch_size=batch_size, 
                    shuffle=False
    )

    return dataset, loader

def compute_riqa_scores():
    model = load_model()
    
    fields = ['21015', '21016']
    for field_id in fields:
        dataset, loader = get_dataloader(field_id)

        #with open(f'./data/preproc/riqa_scores_{field_id}.csv', 'a+') as scores:
        #    scores_writer = csv.writer(scores)
        #    scores_writer.writerow(
        #            ['participant_id', 'good', 'usable', 'reject']
        #    )
            
        for batch, images in enumerate(loader):
            with open(f'./progress_{field_id}.log', 'w') as log_file:
                log_file.write(f'Predicting Batch: {batch+1}/{len(loader)}\n')
            images_rgb = images.to(device)
            hsv_img = cv2.cvtColor(images_rgb, cv2.COLOR_BGR2HSV)
            lab_img = cv2.cvtColor(images_rgb, cv2.COLOR_BGR2LAB)
            images_size = images.size()[0]
            y_hat = model(images_rgb, hsv_img, lab_img)
            item_idxs = [i+batch*images_size for i in range(images_size)]
            participants = dataset.get_items(item_idxs, images_size)
            preds = np.concatenate(
                        (participants, y_hat[-1].cpu().detach().numpy()), 
                        axis=1
            )
            
            with open(f'./data/preproc/riqa_scores_{field_id}.csv', 'a+') as scores:
                scores_writer = csv.writer(scores)
                scores_writer.writerows(preds)

            # save some randomly selected samples for sanity check purposes
            if np.random.uniform(0,1,1)[0] < 0.001:
                for i, img in enumerate(images):
                    cv2.imwrite(
                        f'./data/interim/{participants[i][0]}_{field_id}.png', 
                        images[i].cpu().permute(2,1,0).detach().numpy()
                    )

        def comp_probs(scores_list):
            probs = [score/sum(scores_list) for score in scores_list]
            return probs

        riqa_scores = pd.read_csv(f'./data/preproc/riqa_scores_{field_id}.csv').values
        riqa_scores_adj = []
        for line in riqa_scores:
            probs = comp_probs(line[1:])
            probs.insert(0, line[0])
            riqa_scores_adj.append(probs)

        scores_adj = pd.DataFrame(data=riqa_scores_adj,
                            columns=['participant_id', 'good', 'usable', 'reject'])

        scores_adj.to_csv(f'./data/preproc/riqa_scores_{field_id}_norm.csv', index=False)


if __name__ == '__main__':
    compute_riqa_scores()
    
                