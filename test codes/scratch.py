import numpy as np
import pandas as pd



data = np.load('/Users/ayanbask/PycharmProjects/VAE/results/latent_vectors_t1.npy', allow_pickle=True)
labels = np.load('/Users/ayanbask/PycharmProjects/VAE/results/labels_t1.npy', allow_pickle=True)
labels = labels.astype('float32')

new_data = np.concatenate([data, labels], axis=1)

# print(data[0])
# print(type(data[0][0]))
# print(labels[0])
# print(type(labels[0][0]))

# print(new_data[0])
print(np.concatenate([data[0], labels[0]]))

