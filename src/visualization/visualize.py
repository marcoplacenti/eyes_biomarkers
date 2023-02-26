import joblib
import os
import numpy as np
import umap
import pandas as pd
import matplotlib.pyplot as plt

emb_dir = './data/embeddings/inception/'
emb_files = os.listdir(emb_dir)

participants_ids = [int(emb.split('_')[0]) for emb in emb_files]

info = pd.read_csv('./data/meta/train_set.csv')
info = info[info['participant_id'].isin(participants_ids)]
info = info[['participant_id', 'gender', 'sbp', 'side']]

embeddings = np.array([joblib.load(emb_dir+emb) for emb in emb_files])
sbp_labels, sex_labels = [], []
for part_id in participants_ids:
    row = info[info['participant_id'] == part_id].values[0]
    if row[2] >= 140:
        sbp_labels.append(1)
    else:
        sbp_labels.append(0)
    sex_labels.append(row[1])

print('Executing UMAP')
reducer = umap.UMAP()
reduced_emb = reducer.fit_transform(embeddings)#, n_components=3)

#umap.plot.points(reduced_emb, labels=sbp_labels, color_key_cmap='Paired', background='black')

fig, ax = plt.subplots(figsize=(12, 10))
color = sbp_labels
plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], label=color, c=color)
plt.setp(ax, xticks=[], yticks=[])
plt.title("Data embedded into two dimensions by UMAP", fontsize=18)

plt.savefig(f"./figures/umap_sbp.png")

fig, ax = plt.subplots(figsize=(12, 10))
color = sex_labels
plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], label=color, c=color)
plt.setp(ax, xticks=[], yticks=[])
plt.title("Data embedded into two dimensions by UMAP", fontsize=18)

plt.savefig(f"./figures/umap_sex.png")
