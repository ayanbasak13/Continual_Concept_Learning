import json
import numpy as np
import runners.prior_factory as prior


# means = json.load(open("/Users/ayanbask/PycharmProjects/VAE/Autoencoder/class_means.json"))
# stds = json.load(open("/Users/ayanbask/PycharmProjects/VAE/Autoencoder/class_stds.json"))


def generate_stratified_samples(batch_size, z_dim):
    means = json.load(open("/Users/ayanbask/PycharmProjects/VAE/Autoencoder/class_means.json"))
    stds = json.load(open("/Users/ayanbask/PycharmProjects/VAE/Autoencoder/class_stds.json"))

    l = []

    l1 = list(means.keys())
    l2 = list(stds.keys())

    mu_ = means[l1[0]]
    sigma_ = stds[l2[0]]

    arr = prior.gaussian(10, z_dim, mean=mu_, var=np.square(sigma_))



    for i in range(1, 10):
        mu = means[l1[i]]
        sigma = stds[l2[i]]
        # print(len(mu))
        # print(len(sigma))
        sample = prior.gaussian(10, z_dim, mean=mu, var=np.square(sigma))
        # print(sample)
        # print(type(sample))

        arr = np.concatenate((arr, sample), axis = 0)
        print('Shape  ', arr.shape)

    return arr



a = generate_stratified_samples(100, 20)

# print(a)
print(type(a))
print(a.shape)



# import tensorflow as tf
#
# eof = tf.one_hot([5] * 32, depth=5+1)
# print(eof)
