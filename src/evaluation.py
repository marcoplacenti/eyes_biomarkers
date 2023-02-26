import os
import csv
import math
import numpy as np
import pandas as pd
import joblib
from collections import Counter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import precision_recall_fscore_support

import umap
import matplotlib.pyplot as plt

MULTISTAGE_HYPERTENSION = True

def evaluate(model_name, reduced_emb, part_ids, x, y, task, reduce_mode):
    
    print(f"Metrics for embeddings {model_emb} for {task} prediction...")
    print(f"{task} class distribution: ", Counter(y))

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(x, y)

    print("DB index: ", davies_bouldin_score(x, y))

    print("KNN score detection: ", neigh.score(x, y))

    preds = list(neigh.predict(x))
    preds_prob = neigh.predict_proba(x)[:, 1]

    if not MULTISTAGE_HYPERTENSION:
        inference_dir = './data/inference'
        if not os.path.exists(inference_dir+f'/{model_name}'):
            os.makedirs(inference_dir+f'/{model_name}')

        with open(inference_dir+f'/{model_name}/{task}_preds.csv', 'w') as inf:
            scores_writer = csv.writer(inf)
            inf.write('ParticipantID,GroundTruth,Prediction,Probability\n')
            for i, pred in enumerate(preds):
                part_id = part_ids[i]
                prob = preds_prob[i]
                inf.write(f'{part_id},{y[i]},{pred},{prob}\n')
    
    if not MULTISTAGE_HYPERTENSION:
        color = []
        label = []
        for i, pred in enumerate(preds):
            if pred == 1 and y[i] == 1:
                color.append('limegreen')
                label.append('tp')
            if pred == 0 and y[i] == 0:
                color.append('darkcyan')
                label.append('tn')
            if pred > y[i]:
                color.append('royalblue')
                label.append('fp')
            if pred < y[i]:
                color.append('blueviolet')
                label.append('fn')

        plt.clf()
        fig, ax = plt.subplots(figsize=(12, 10))
        color = color
        plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], s=10, label=label, c=color)
        plt.title(f"UMAP-{model_emb} for {task} classification", fontsize=22)
        plt.savefig(f"./figures/umap_{task}_{model_emb}_{reduce_mode}_colorful.png")

    else:
        plt.clf()
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], s=10, label=preds, c=preds, cmap='seismic')
        plt.title(f"UMAP-{model_emb} for {task} classification", fontsize=22)
        plt.savefig(f"./figures/umap_{task}_{model_emb}_{reduce_mode}_colorful.png")
        exit()
    """
    print("Confusion Matrix: ", confusion_matrix(y, preds))
    fpr, tpr, thresholds = roc_curve(y, preds_prob)
    auc = roc_auc_score(y, preds_prob)

    plt.clf()
    plt.rcParams.update({'font.size': 22})
    plt.plot(fpr, tpr, 'b', label=f'AUC = {round(auc, 3)}')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    #plt.xlim([0, 1])
    #plt.ylim([0, 1])
    plt.title(f"ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(f"./figures/ROC_{task}_{model_emb}_{reduce_mode}.png")
    """
if __name__ == '__main__':
    cand_pat_info = pd.read_csv('./data/meta/candidate_patients_info.csv')
    all_cand = np.unique(cand_pat_info['f.eid'].values)

    post_qc_scan_instance = pd.read_csv('./data/meta/post_qc_scan_instance_set.csv')
    all_part_instance = np.unique(post_qc_scan_instance['participant_id'].values)
    missing = [cand for cand in all_cand if cand not in all_part_instance]

    train_set = pd.read_csv('./data/meta/train_set.csv')
    test_set = pd.read_csv('./data/meta/test_set.csv')
    val_set = pd.read_csv('./data/meta/val_set.csv')
    full_set = pd.concat([train_set, test_set, val_set])

    discarded = full_set.loc[(full_set['left_eye_crop'] == '[]') | (full_set['right_eye_crop'] == '[]')]
    discarded = np.unique(discarded['participant_id'].values)

    full_set = full_set.loc[full_set['left_eye_crop'] != '[]']
    full_set = full_set.loc[full_set['right_eye_crop'] != '[]']

    full_set = full_set[['participant_id', 'gender', 'sbp']]
    full_set.drop_duplicates(inplace=True)

    full_set.dropna(inplace=True)

    if MULTISTAGE_HYPERTENSION:
        stages = {range(0, 119): 0, range(120, 129): 1, range(130, 139): 2, 
                range(140, 179): 3, range(180, 300): 4}
        full_set['sbp'] = full_set['sbp'].apply(lambda x: next((v for k, v in stages.items() if x in k), 0))
    else:
        full_set['sbp'] = (full_set['sbp'] >= 140).astype(float)
    
    lookup_dict = {str(int(row[0])): {'gender': row[1], 'ht': row[2]}
                    for row in full_set.values}

    #models_emb = os.listdir('./data/semiold_embs/')
    models_emb = ['InceptionV3_Triplet_basic_128_224', 'InceptionV3_Triplet_dataaug_128_299',
                    'AE_PerceptualLoss_basic_128_224', 'AE_PerceptualLoss_dataaug_128_224',
                    'AE_SSIMLoss_basic_128_224', 'AE_SSIMLoss_dataaug_128_224',
                    'MoCo_Contrastive_basic_128_224', 'MoCo_Contrastive_dataaug_128_224']
                    
    for model_emb in models_emb:
        emb_dir = './data/semiold_embs/'+model_emb

        embeddings = sorted(os.listdir(emb_dir))

        part_ids = []
        x = []
        y_gender = []
        y_ht = []
        y_side = []
        x_mean_dict, x_mean = {}, []
        y_mean_gender, y_mean_ht, y_mean_side = [], [], []
        for i, f in enumerate(embeddings[:20000]):
            try:
                part_id = str(f.split('_')[0])
                y_gender.append(lookup_dict[part_id]['gender'])
                y_ht.append(lookup_dict[part_id]['ht'])
                side = (0 if str(f.split('_')[1]) == '21015' else 1)
                y_side.append(side)
                emb = joblib.load(f'{emb_dir}/{f}')
                x.append(emb)
                if part_id in list(x_mean_dict.keys()):
                    part_ids.append(part_id)
                    x_mean.append((x_mean_dict[part_id] + emb ) / 2) 
                    y_mean_gender.append(lookup_dict[part_id]['gender'])
                    y_mean_ht.append(lookup_dict[part_id]['ht'])
                    y_mean_side.append(side)
                else:
                    x_mean_dict[part_id] = emb

            except KeyError:
                continue

        print("Computing UMAP...")
        reducer = umap.UMAP()
        reduced_emb = reducer.fit_transform(x_mean)#, n_components=3)
        
        #evaluate(model_emb, reduced_emb, part_ids, x_mean, y_mean_gender, 'sex', 'avg')
        if MULTISTAGE_HYPERTENSION:
            evaluate(model_emb, reduced_emb, part_ids, x_mean, y_mean_ht, 'multistage_hypertension', 'avg')
        else:
            evaluate(model_emb, reduced_emb, part_ids, x_mean, y_mean_ht, 'hypertension', 'avg')
        #reducer = umap.UMAP()
        #reduced_emb = reducer.fit_transform(x)#, n_components=3)
        #evaluate(model_emb, reduced_emb, x, y_gender, 'sex', 'all')
        #evaluate(model_emb, reduced_emb, x, y_ht, 'hypertension', 'all')

        