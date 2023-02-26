import os
import pandas as pd
import joblib
from collections import Counter

from sklearn.metrics import silhouette_score, davies_bouldin_score

from sklearn.neighbors import KNeighborsClassifier

import umap
import matplotlib.pyplot as plt


"""
There are many different clustering metrics that can be used to evaluate the quality of a clustering, and the time complexity of these metrics can vary significantly. Here are some common clustering metrics and their time complexity:

Silhouette score: O(n^2)
Adjusted Rand index: O(n^2)
Adjusted Mutual Information (AMI): O(n^2)
Homogeneity, completeness, and V-measure: O(n^2)
Calinski-Harabasz (CH) index: O(n^2)
As you can see, many of the common clustering metrics have an O(n^2) time complexity, which means they can be computationally expensive when applied to large datasets. In these cases, it may be necessary to use a faster, approximative version of the metric or to use a different metric that has a lower time complexity.

One metric that has a lower time complexity than the others listed above is the Davies-Bouldin (DB) index, which has a time complexity of O(nk), where k is the number of clusters. The DB index is based on the average distance between the centers of clusters, which can be calculated more efficiently than the distances between all pairs of samples that are used by some of the other metrics.
"""

train_set = pd.read_csv('./data/meta/train_set.csv')
test_set = pd.read_csv('./data/meta/test_set.csv')
val_set = pd.read_csv('./data/meta/val_set.csv')
full_set = pd.concat([train_set, test_set, val_set])

full_set = full_set[['participant_id', 'gender', 'sbp']]
full_set['sbp'] = (full_set['sbp'] > 140).astype(float)
full_set.drop_duplicates(inplace=True)

lookup_dict = {str(int(row[0])): {'gender': row[1], 'ht': row[2]}
                for row in full_set.values}


models_emb = ['MoCo_Contrastive']#, 'AE_PerceptualLoss', 'AE_SSIMLoss', 'InceptionV3_Triplet']
for model_emb in models_emb:
    emb_dir = './data/embeddings/'+model_emb #AE_PerceptualLoss' # MoCo_Contrastive

    embeddings = os.listdir(emb_dir)
    embeddings = [emb for emb in embeddings if emb.endswith('_0.emb')]

    x = []
    y_gender = []
    y_ht = []
    for f in embeddings:
        part_id = f.split('_')[0]
        y_gender.append(lookup_dict[part_id]['gender'])
        y_ht.append(lookup_dict[part_id]['ht'])
        emb = joblib.load(f'{emb_dir}/{f}')
        #emb = emb[0]
        x.append(emb)

    print("Computing UMAP...")
    reducer = umap.UMAP()
    reduced_emb = reducer.fit_transform(x)#, n_components=3)

    fig, ax = plt.subplots(figsize=(12, 10))
    color = y_gender
    plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], label=color, c=color)
    plt.setp(ax, xticks=[], yticks=[])
    plt.title("Data embedded into two dimensions by UMAP", fontsize=18)

    plt.savefig(f"./figures/umap_sex_{model_emb}.png")

    fig, ax = plt.subplots(figsize=(12, 10))
    color = y_ht
    plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], label=color, c=color)
    plt.setp(ax, xticks=[], yticks=[])
    plt.title("Data embedded into two dimensions by UMAP", fontsize=18)

    plt.savefig(f"./figures/umap_ht_{model_emb}.png")

    print(f"Metrics for embeddings {model_emb}")
    print("Gender class distribution: ", Counter(y_gender))
    print("Hypertension class distribution: ", Counter(y_ht))

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(x, y_gender)

    print("KNN score for gender detection: ", neigh.score(x, y_gender))

    print("Computing score")
    score = davies_bouldin_score(x, y_gender)
    print("Davies Bouldin Score for gender clustering: ", score)


    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(x, y_ht)

    print("KNN score for hypertension status detection: ", neigh.score(x, y_ht))

    print("Computing score")
    score = davies_bouldin_score(x, y_ht)
    print("Davies Bouldin Score for hypertension status clustering: ", score)