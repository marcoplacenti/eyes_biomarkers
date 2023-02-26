import pandas as pd
from math import exp
import numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

test_set = pd.read_csv('./data/meta/test_set.csv')
val_set = pd.read_csv('./data/meta/val_set.csv')
train_set = pd.read_csv('./data/meta/train_set.csv')

data = pd.concat([test_set, val_set, train_set])

instances = pd.read_csv('./data/meta/post_qc_scan_instance_set.csv')

# how many instances (how were they filtered out in the first place? inspect zots' code)
print(f'Number of patients: {len(instances)}')

# how many missing left and right eyes
print(f"Number of missing left: {len(instances[instances['left_eye_crop'] == '[]'])}")
print(f"Number of missing right: {len(instances[instances['right_eye_crop'] == '[]'])}")

# how many missing both - should be 0
print(f"Number of missing both: {len(instances[(instances['right_eye_crop'] == '[]') & (instances['left_eye_crop'] == '[]')])}")

part_ids = instances['participant_id'].values

filtered_data = data[data['participant_id'].isin(part_ids)]
sex_df = filtered_data[['participant_id', 'gender']].drop_duplicates()

# count sexes
print(f"Sexes: {Counter(sex_df['gender'].values)}")

# patients with no sex information
filtered_data = sex_df[~sex_df['participant_id'].isin(part_ids)]
print(f'Number of patients in gender file but not in instances file: {len(filtered_data)}')
sex_part_ids = sex_df['participant_id'].values
filtered_data = instances[~instances['participant_id'].isin(sex_part_ids)]
print(f'Number of patients in instances file but not in gender file: {len(filtered_data)}')

# count sexes of missing left and sexes of missing right
missing_left = instances.loc[instances['left_eye_crop'] == '[]']
missing_left_ids = missing_left['participant_id'].values
print(f"Sexes of missing left: {Counter(sex_df[sex_df['participant_id'].isin(missing_left_ids)]['gender'].values)}")

missing_right = instances.loc[instances['right_eye_crop'] == '[]']
missing_right_ids = missing_right['participant_id'].values
print(f"Sexes of missing right: {Counter(sex_df[sex_df['participant_id'].isin(missing_right_ids)]['gender'].values)}")


# mcf net outcome analysis
def comp_probs(scores_list):
    probs = [score/sum(scores_list) for score in scores_list]
    return probs

riqa_scores_left = pd.read_csv('./data/preproc/riqa_scores_21015.csv').values
riqa_scores_left_adj = []
for line in riqa_scores_left:
    probs = comp_probs(line[1:])
    probs.insert(0, line[0])
    riqa_scores_left_adj.append(probs)

preds = [np.argmax(item[1:]) for item in riqa_scores_left_adj]
print(f"Scores left: {Counter(preds)}")

riqa_scores_right = pd.read_csv('./data/preproc/riqa_scores_21016.csv').values
riqa_scores_right_adj = []
for line in riqa_scores_right:
    probs = comp_probs(line[1:])
    probs.insert(0, line[0])
    riqa_scores_right_adj.append(probs)

preds = [np.argmax(item[1:]) for item in riqa_scores_right_adj]
print(f"Scores right: {Counter(preds)}")

# count how many left and how many right (before and after preprocessing)



# thresholds
thresh_vals = np.arange(0.05, 1, 0.05)
thresh_counter = {
    'Good': [0 for val in thresh_vals],
    'Usable': [0 for val in thresh_vals],
    'Reject': [0 for val in thresh_vals]
}
class_mapping = {0: 'Good', 1: 'Usable', 2: 'Reject'}

for i, t in enumerate(thresh_vals):
    for line in riqa_scores_left:
        probs = line[1:]
        argmax_idx = np.argmax(probs)
        if probs[argmax_idx] > t:
            thresh_counter[class_mapping[argmax_idx]][i] += 1

df = pd.DataFrame(thresh_counter,
                  index=thresh_vals)
 
# create stacked bar chart for monthly temperatures
df.plot(kind='bar', stacked=True, color=['darkgreen', 'skyblue', 'darkred'])
 
# labels for x & y axis
plt.xlabel('Thresholds')
plt.ylabel('Predictions')
 
# title of plot
plt.title('Quality prediction for 20 thresholds')

plt.savefig('./figures/thresholds.png')

for i, t in enumerate(thresh_vals):
    qty_pos = thresh_counter['Good'][i]+thresh_counter['Usable'][i]
    qty_neg = thresh_counter['Reject'][i]

    print(f'{t}: {qty_pos}, {qty_neg}, {qty_pos/(qty_pos+qty_neg)}, {qty_pos/len(instances)}')