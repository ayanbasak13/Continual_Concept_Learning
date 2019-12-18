import pandas as pd
import numpy as np
import json


class_means = {}
class_stds = {}


means = np.load('/Users/ayanbask/PycharmProjects/VAE/Autoencoder/means_t1.npy', allow_pickle=True)
stds = np.load('/Users/ayanbask/PycharmProjects/VAE/Autoencoder/stds_t1.npy', allow_pickle=True)
labels = np.load('/Users/ayanbask/PycharmProjects/VAE/Autoencoder/labels_t1.npy', allow_pickle=True)
lab = [-1]*len(labels)

print(means.shape)
print(stds.shape)
print(labels.shape)
# print(labels[0])
for i,l in enumerate(labels) :
    for j in range(0, len(l)) :
        if(l[j]==1):
            lab[i]=j


# print(labels[1], lab[1])

df=pd.DataFrame()

df["means"] = list(means)
df["stds"] = list(stds)
df["labels"] = list(lab)


grouped = df.groupby('labels').groups

for group in grouped :
    means_sum=0
    stds_sum=0
    idx = grouped[group].tolist()
    for i in idx :
        means_sum+=means[i]
        stds_sum+=stds[i]

    class_means[group] = (means_sum/len(idx))
    class_stds[group] = stds_sum/len(idx)

for g in class_means :
    class_means[g] = class_means[g].tolist()

for s in class_stds :
    class_stds[s] = class_stds[s].tolist()

print(class_means)
print(class_stds)


json.dump(dict(class_means), open("/Users/ayanbask/PycharmProjects/VAE/Autoencoder/class_means.json", 'w'))
json.dump(dict(class_stds), open("/Users/ayanbask/PycharmProjects/VAE/Autoencoder/class_stds.json", 'w'))

