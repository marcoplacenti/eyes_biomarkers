import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

confusion_matrix_dict = {
    'AE_PerceptualLoss_nonaug_128_224_sex_aggr': np.array([[22892, 6412],[8366,14812]]),
    'AE_PerceptualLoss_nonaug_128_224_ht_aggr': np.array([[20046, 8085],[7485,16866]]),
    'AE_PerceptualLoss_nonaug_128_224_sex_nonaggr': np.array([[45266,13342],[17118,29238]]),
    'AE_PerceptualLoss_nonaug_128_224_ht_nonaggr': np.array([[39851,16411],[15305,33397]]),
    'AE_PerceptualLoss_aug_128_224_sex_aggr': np.array([[22806,6498],[ 8368,14810]]),
    'AE_PerceptualLoss_aug_128_224_ht_aggr': np.array([[20238,7893],[ 7617,16734]]),
    'AE_PerceptualLoss_aug_128_224_sex_nonaggr': np.array([[45072,13536],[17113,29243]]),
    'AE_PerceptualLoss_aug_128_224_ht_nonaggr': np.array([[39870,16392],[15190,33512]]),

    'AE_DSSIMLoss_nonaug_128_224_sex_aggr': np.array([[23753,5551],[ 8939 ,14239]]),
    'AE_DSSIMLoss_nonaug_128_224_ht_aggr': np.array([[20485,7646],[8061,16290]]),
    'AE_DSSIMLoss_nonaug_128_224_sex_nonaggr': np.array([[47430, 11178],[18461, 27895]]),
    'AE_DSSIMLoss_nonaug_128_224_ht_nonaggr': np.array([[41380 ,14882],[16189 ,32513]]),
    'AE_DSSIMLoss_aug_128_224_sex_aggr': np.array([[23662  ,5642],[ 8910, 14268]]),
    'AE_DSSIMLoss_aug_128_224_ht_aggr': np.array([[20201  ,7930], [ 7759, 16592]]),
    'AE_DSSIMLoss_aug_128_224_sex_nonaggr': np.array([[47018 ,11590], [18227 ,28129]]),
    'AE_DSSIMLoss_aug_128_224_ht_nonaggr': np.array([[40986 ,15276], [16080, 32622]]),

    'InceptionV3_Triplet_nonaug_128_299_sex_aggr': np.array([[22705 , 6599],[ 8747 ,14431]]),
    'InceptionV3_Triplet_nonaug_128_299_ht_aggr': np.array([[20999 , 7132],[ 8136 ,16215]]),
    'InceptionV3_Triplet_nonaug_128_299_sex_nonaggr': np.array([[45726 ,12882],[16693 ,29663]]),
    'InceptionV3_Triplet_nonaug_128_299_ht_nonaggr': np.array([[42605, 13657],[15719, 32983]]),
    'InceptionV3_Triplet_aug_128_299_sex_aggr': np.array([[22712,  6592],[ 8849 ,14329]]),
    'InceptionV3_Triplet_aug_128_299_ht_aggr': np.array([[20979 , 7152],[ 8353 ,15998]]),
    'InceptionV3_Triplet_aug_128_299_sex_nonaggr': np.array([[45191, 13417],[17700 ,28656]]),
    'InceptionV3_Triplet_aug_128_299_ht_nonaggr': np.array([[42270 ,13992],[16506, 32196]]),

    'MoCo_Contrastive_nonaug_128_224_sex_aggr': np.array([[22805 , 6499],[ 9413 ,13765]]),
    'MoCo_Contrastive_nonaug_128_224_ht_aggr': np.array([[20867  ,7264],[ 9002 ,15349]]),
    'MoCo_Contrastive_nonaug_128_224_sex_nonaggr': np.array([[45470 ,13138],[18734 ,27622]]),
    'MoCo_Contrastive_nonaug_128_224_ht_nonaggr': np.array([[41166 ,15096],[17597 ,31105]]),
    'MoCo_Contrastive_aug_128_224_sex_aggr': np.array([[22795 , 6509],[ 9270, 13908]]),
    'MoCo_Contrastive_aug_128_224_ht_aggr': np.array([[20944  ,7187],[ 8800 ,15551]]),
    'MoCo_Contrastive_aug_128_224_sex_nonaggr': np.array([[45337 ,13271],[18443 ,27913]]),
    'MoCo_Contrastive_aug_128_224_ht_nonaggr': np.array([[41606 ,14656],[17784 ,30918]]),

}

confusion_matrix_dict = {
    'InceptionV3_Triplet_nonaug_128_299_mlht_aggr': np.array([
                    [ 5525,   850,   946,  2760,    28],
                    [ 1615,  3382,   843,  2572,    26],
                    [ 1714,  1081,  3852,  3057,    29],
                    [ 2750,  1831,  2266, 15632,    32],
                    [  303,   152,   232,   919,    85]])
}

confusion_matrix_dict = {
    'InceptionV3_Triplet_nonaug_128_299_sex_aggr': np.array([[22797,  6507],[ 8950 ,14228]]),
    'InceptionV3_Triplet_nonaug_128_299_ht_aggr': np.array([[21005 , 7126],[ 8245, 16106]])
}


for k in list(confusion_matrix_dict.keys()):
    plt.clf()
    df_cm = pd.DataFrame(confusion_matrix_dict[k], range(2), range(2))

    ax = plt.subplot()
    sn.heatmap(df_cm, annot=True, fmt='d', cmap='BuGn', vmin=0, ax=ax, annot_kws={"size": 18}) # font size
    
    ax.set_xlabel('Predictions', fontsize=12)
    ax.set_ylabel('Ground Truth', fontsize=12)
    plt.title(f'Confusion Matrix {k.split("_")[0]}-{k.split("_")[-2]}-{k.split("_")[2]}', fontsize=12)

    
    if k.split('_')[-2] == 'sex':
        ax.xaxis.set_ticklabels(['female', 'male'])
        ax.yaxis.set_ticklabels(['female', 'male'])
    else:
        ax.xaxis.set_ticklabels(['no ht', 'ht'])
        ax.yaxis.set_ticklabels(['no ht', 'ht'])
    

    plt.savefig(f'./figures/CM_{k}.png')

    tp = confusion_matrix_dict[k][0][0]
    fp = confusion_matrix_dict[k][0][1]
    fn = confusion_matrix_dict[k][1][0]
    tn = confusion_matrix_dict[k][1][1]

    precision = round(tp / (tp+fp), 4)
    recall = round(tp / (tp + fn), 4)
    f1_score = round(2 * (precision * recall) / (precision + recall), 4)

    print(k, precision, recall, f1_score)