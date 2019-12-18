import os
import gzip
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import  Image




# def extract_data(filename, num_data, head_size, data_size):
#     with gzip.open(filename) as bytestream:
#         bytestream.read(head_size)
#         buf = bytestream.read(data_size * num_data)
#         data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
#     return data
#
#
# data = extract_data('/Users/ayanbask/PycharmProjects/VAE/data/mnist/train-images-idx3-ubyte.gz', 1, 16, 28 * 28)
# # trX = data.reshape((60000, 28, 28, 1))
# print(data)
# print('Data Shape ', data.shape)
# pixels = np.array(data, dtype='uint8')
#
# # Reshape the array into 28 x 28 array (2-dimensional array)
# pixels = pixels.reshape((28, 28))
#
#
#
# data_lab = extract_data('/Users/ayanbask/PycharmProjects/VAE/data/mnist/train-labels-idx1-ubyte.gz', 1, 8, 1)
# # trY = data.reshape((60000))
# trY = data_lab.reshape((1))
# # Plot
# plt.title('Label is {label}'.format(label=trY))
# plt.imshow(pixels, cmap='gray')
# plt.show()
#
#
#
# new_im = Image.fromarray(pixels)
# new_im.save("numpy_altered_sample.png")



# Reading the input image
img = cv2.imread('numpy_altered_sample.png', 0)

# Taking a matrix of size 5 as the kernel
kernel = np.ones((5, 5), np.uint8)

# The first parameter is the original image,
# kernel is the matrix with which image is
# convolved and third parameter is the number
# of iterations, which will determine how much
# you want to erode/dilate a given image.
img_erosion = cv2.erode(img, kernel, iterations=1)
img_dilation = cv2.dilate(img, kernel, iterations=1)

cv2.imshow('Input', img)
cv2.imshow('Erosion', img_erosion)
cv2.imshow('Dilation', img_dilation)

cv2.waitKey(0)






