"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""
from __future__ import division
import math
import random
import pprint
import scipy.misc
import pandas as pd
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os, gzip
import cv2
import matplotlib.pyplot as plt
from PIL import  Image

import tensorflow as tf
import tensorflow.contrib.slim as slim

def load_mnist(dataset_name):
    print("TRAINING ON MNIST")
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data('/Users/ayanbask/PycharmProjects/VAE/data/mnist/train-images-idx3-ubyte.gz', 36000, 16, 28 * 28)
    # trX = data.reshape((60000, 28, 28, 1))
    trX = data.reshape((36000, 28, 28, 1))

    val_data = extract_data('/Users/ayanbask/PycharmProjects/VAE/data/mnist/train-images-idx3-ubyte.gz', 24000, 16+(36000*28*28), 28 * 28)
    valX = val_data.reshape((24000, 28, 28, 1))

    data = extract_data('/Users/ayanbask/PycharmProjects/VAE/data/mnist/train-labels-idx1-ubyte.gz', 36000, 8, 1)
    # trY = data.reshape((60000))
    trY = data.reshape((36000))

    val_data = extract_data('/Users/ayanbask/PycharmProjects/VAE/data/mnist/train-labels-idx1-ubyte.gz', 24000, 8+36000, 1)
    # trY = data.reshape((60000))
    valY = val_data.reshape((24000))

    data = extract_data('/Users/ayanbask/PycharmProjects/VAE/data/mnist/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    # teX = data.reshape((10000, 28, 28, 1))
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data('/Users/ayanbask/PycharmProjects/VAE/data/mnist/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    # teY = data.reshape((10000))
    teY = data.reshape((10000))



    trY = np.asarray(trY)
    teY = np.asarray(teY)
    valY = np.asarray(valY)

    # X = np.concatenate((trX, teX), axis=0)
    # y = np.concatenate((trY, teY), axis=0).astype(np.int)

    trY = trY.astype(np.int)
    valY = valY.astype(np.int)
    teY = teY.astype(np.int)

    # seed = 547
    # np.random.seed(seed)
    # np.random.shuffle(trX)
    # np.random.shuffle(teX)
    # np.random.shuffle(valX)
    # np.random.seed(seed)
    # np.random.shuffle(trY)
    # np.random.shuffle(teY)
    # np.random.shuffle(valY)





    pixels = np.array(np.squeeze(trX[0]), dtype='uint8')
    #
    # new_im = Image.fromarray(pixels)
    # new_im.save("numpy_altered_sample.png")
    #
    # # Reading the input image
    # img = cv2.imread('numpy_altered_sample.png', 0)
    #
    # # Taking a matrix of size 5 as the kernel
    # kernel = np.ones((5, 5), np.uint8)
    #
    # # cv2.imshow('Input', img)
    #
    #
    # cv2.waitKey(0)
    #
    #
    # pixels1 = np.array(np.squeeze(valX[0]), dtype='uint8')
    #
    # new_im = Image.fromarray(pixels1)
    # new_im.save("numpy_altered_sample1.png")
    #
    # # Reading the input image
    # img = cv2.imread('numpy_altered_sample1.png', 0)
    #
    # # Taking a matrix of size 5 as the kernel
    # kernel = np.ones((5, 5), np.uint8)
    #
    # # cv2.imshow('Input', img)

    # cv2.waitKey(0)

    # exit()

    y_vec_tr = np.zeros((len(trY), 10), dtype=np.float)
    for i, label in enumerate(trY):
        # print(i,label)
        y_vec_tr[i, trY[i]] = 1.0

    y_vec_val = np.zeros((len(valY), 10), dtype=np.float)
    for i, label in enumerate(valY):
        # print(i, label)
        y_vec_val[i, valY[i]] = 1.0
    # exit()

    y_vec_te = np.zeros((len(teY), 10), dtype=np.float)
    for i, label in enumerate(teY):
        y_vec_te[i, teY[i]] = 1.0


    """
    for image,label in zip(valX,valY):
        print(label)
        pixels = np.array(np.squeeze(image), dtype='uint8')
        new_im = Image.fromarray(pixels)
        new_im.save("numpy_altered_sample.png")
        #
        # # Reading the input image
        img = cv2.imread('numpy_altered_sample.png', 0)
        cv2.imshow("hh",img)
        cv2.waitKey(0)
    """


    print(trX.shape)
    print(trY.shape)
    print(teX.shape)
    print(teY.shape)
    print(valX.shape)
    print(valY.shape)
    print(set(trY))


    return trX / 255., y_vec_tr, valX / 255., y_vec_val, teX / 255., y_vec_te





def load_e_mnist(dataset_name):
    print("TRAINING ON E-MNIST")
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = pd.read_csv("/Users/ayanbask/PycharmProjects/VAE/data/emnist/emnist-letters-train.csv")
    data = shuffle(data)
    print("Len data ", len(data))
    print(data.shape)

    val_data = data[36000:60000]
    unsupervised_data = data[60000:96000]
    print("Len val data ", len(val_data))
    data = data[:36000]
    print("Len data ", len(data))
    print(type(data))
    print("Len unsupervised data ", len(unsupervised_data))

    trX = data.values[:,1:]
    trX = trX.reshape((trX.shape[0], 28, 28, 1))
    valX = val_data.values[:,1:]
    valX = valX.reshape((valX.shape[0], 28, 28, 1))
    unsX = unsupervised_data.values[:,1:]
    unsX = unsX.reshape((unsX.shape[0], 28, 28, 1))


    # data = extract_data('/Users/ayanbask/PycharmProjects/VAE/data/mnist/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.values[:,0]

    trY = trY.reshape((trX.shape[0]))
    print('Set ', set(trY))

    valY = val_data.values[:,0]

    valY = valY.reshape((valX.shape[0]))
    print('Set ', set(valY))


    unsY = unsupervised_data.values[:,0]

    unsY = unsY.reshape((unsX.shape[0]))

    data = pd.read_csv("//Users/ayanbask/PycharmProjects/VAE/data/emnist/emnist-letters-test.csv")
    data = shuffle(data)
    data = data[:10000]
    teX = data.values[:,1:]
    teX = teX.reshape((teX.shape[0], 28, 28, 1))

    # data = extract_data('/Users/ayanbask/PycharmProjects/VAE/data/mnist/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.values[:,0]
    teY = teY.reshape((teX.shape[0]))

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    valY = np.asarray(valY)
    unsY = np.asarray(unsY)

    print("Len of val Y", len(valY))

    # X = np.concatenate((trX, teX), axis=0)
    # y = np.concatenate((trY, teY), axis=0).astype(np.int)



    # for image,label in zip(valX,valY):
    #     print(label)
    #     pixels = np.array(np.squeeze(image), dtype='uint8')
    #     new_im = Image.fromarray(pixels)
    #     new_im.save("numpy_altered_sample.png")
    #     #
    #     # # Reading the input image
    #     img = cv2.imread('numpy_altered_sample.png', 0)
    #     cv2.imshow("hh",img)
    #     cv2.waitKey(0)


    # pixels = np.array(np.squeeze(X[0]), dtype='uint8')
    #
    # new_im = Image.fromarray(pixels)
    # new_im.save("numpy_altered_sample.png")
    #
    # # Reading the input image
    # img = cv2.imread('numpy_altered_sample.png', 0)
    #
    # # Taking a matrix of size 5 as the kernel
    # kernel = np.ones((5, 5), np.uint8)
    #
    # cv2.imshow('Input', img)
    #
    # cv2.waitKey(0)



    for i, lab in enumerate(trY) :
        trY[i] = trY[i] - 1
    print('TYPE OF train LABEL', type(trY[0]))

    for i, lab in enumerate(valY) :

        valY[i] = valY[i] - 1
    # print('TYPE OF val LABEL', type(valY[0]))

    for i, lab in enumerate(unsY) :

        unsY[i] = unsY[i] - 1

    for i, lab in enumerate(teY) :
        teY[i] = teY[i] - 1
    print('TYPE OF test LABEL', type(teY[0]))


    print(trX.shape)
    print(trY.shape)
    print(teX.shape)
    print(teY.shape)
    print(valX.shape)
    print(valY.shape)
    # print(set(y))


    # seed = 547
    # np.random.seed(seed)
    # np.random.shuffle(trX)
    # np.random.shuffle(valX)
    # np.random.shuffle(teX)
    # np.random.seed(seed)
    # np.random.shuffle(trY)
    # np.random.shuffle(valY)
    # np.random.shuffle(teY)

    # y_vec = np.zeros((len(y), 47), dtype=np.float)
    y_vec_tr = np.zeros((len(trY), 26), dtype=np.float)
    for i, label in enumerate(trY):
        y_vec_tr[i, trY[i]] = 1.0

    y_vec_val = np.zeros((len(valY), 26), dtype=np.float)
    for i, label in enumerate(valY):
        y_vec_val[i, valY[i]] = 1.0

    y_vec_uns = np.zeros((len(unsY), 26), dtype=np.float)
    for i, label in enumerate(unsY):
        y_vec_uns[i, unsY[i]] = 1.0

    y_vec_te = np.zeros((len(teY), 26), dtype=np.float)
    for i, label in enumerate(teY):
        y_vec_te[i, teY[i]] = 1.0

    return trX / 255., y_vec_tr, valX / 255., y_vec_val, teX / 255., y_vec_te, unsX / 255., y_vec_uns





def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, grayscale = False):
    if (grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

""" Drawing Tools """
# borrowed from https://github.com/ykwon0407/variational_autoencoder/blob/master/variational_bayes.ipynb
def save_scattered_image(z, id, z_range_x, z_range_y, name='scattered_image.jpg'):
    N = 10
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    axes.set_xlim([-z_range_x, z_range_x])
    axes.set_ylim([-z_range_y, z_range_y])
    plt.grid(True)
    plt.savefig(name)

# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)





