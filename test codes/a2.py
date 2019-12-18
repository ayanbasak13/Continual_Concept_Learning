import numpy as np
import gzip


def extract_data(filename, num_data, head_size, data_size):
    with gzip.open(filename) as bytestream:
        bytestream.read(head_size)
        buf = bytestream.read(data_size * num_data)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
    return data




val_data = extract_data('/Users/ayanbask/PycharmProjects/VAE/data/mnist/train-images-idx3-ubyte.gz', 24000, 16 + 36000,
                        28 * 28)
valX = val_data.reshape((24000, 28, 28, 1))

print(valX.shape)