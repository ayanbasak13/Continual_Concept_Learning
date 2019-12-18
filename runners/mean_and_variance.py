import numpy as np
import runners.prior_factory as prior
vectors = np.load('/Users/ayanbask/PycharmProjects/VAE/Autoencoder/auto_vectors.npy', allow_pickle=True)

mean = np.mean(vectors)
sigma = np.std(vectors)


print(mean, sigma)


batch_z = prior.gaussian(32, 20, mean=mean, var=np.square(sigma))

print(batch_z[0])
print(len(batch_z))