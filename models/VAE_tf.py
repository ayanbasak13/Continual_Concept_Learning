#-*- coding: utf-8 -*-
# from __future__ import division
import os
import time
import json
import cv2
import tensorflow as tf
import numpy as np

from runners.ops import *
from runners.utils import *
from runners.wassertein import sliced_wasserstein_distance
from scipy.stats import norm
import matplotlib.pyplot as plt


import runners.prior_factory as prior




dir_repeats = 4
dirs_per_repeat = 128



def sliced_wasserstein_tf(A, B, dir_repeats, dirs_per_repeat):

    assert A.shape == B.shape
    print(A.shape, B.shape)                                             # (neighborhood, descriptor_component)
    A = tf.reshape(A, shape = [-1, 28, 28])
    B = tf.reshape(B, shape = [-1, 28, 28])
    assert A.shape == B.shape
    print(A.shape, B.shape)
    # exit()

    results = []
    for repeat in range(dir_repeats):

        with tf.variable_scope('v1') as scope:
            try:
                dirs = tf.get_variable(name='v1', shape=[A.shape[1], dirs_per_repeat], initializer=tf.initializers.random_normal)            # (descriptor_component, direction)
                # dirs = tf.random_normal((A.shape[1], dirs_per_repeat), seed=0)            # (descriptor_component, direction)
                dirs /= tf.sqrt(tf.reduce_sum(tf.square(dirs), axis=0, keepdims=True))                                                   # normalize descriptor components for each direction
                dirs = tf.cast(dirs, tf.float32)
                projA = tf.matmul(A, dirs)                                                                                        # (neighborhood, direction)
                projB = tf.matmul(B, dirs)
                projA = tf.sort(projA, axis=0)                                                                                    # sort neighborhood projections for each direction
                projB = tf.sort(projB, axis=0)
                dists = tf.abs(projA - projB)                                                                                     # pointwise wasserstein distances
                results.append(tf.reduce_mean(dists))                                                                                    # average over neighborhoods and directions

            except ValueError:
                scope.reuse_variables()
                dirs = tf.get_variable(name='v1', shape=[A.shape[1], dirs_per_repeat], initializer=tf.initializers.random_normal)
                # dirs = tf.random_normal((A.shape[1], dirs_per_repeat), seed=0)
                dirs /= tf.sqrt(tf.reduce_sum(tf.square(dirs), axis=0, keepdims=True))
                dirs = tf.cast(dirs, tf.float32)
                projA = tf.matmul(A, dirs)
                projB = tf.matmul(B, dirs)
                projA = tf.sort(projA, axis=0)
                projB = tf.sort(projB, axis=0)
                dists = tf.abs(projA - projB)
                results.append(tf.reduce_mean(dists))


    return tf.reduce_mean(results)



def generate_stratified_samples(batch_size, z_dim):
    means = json.load(open("/Users/ayanbask/PycharmProjects/VAE/Autoencoder/class_means_t1.json"))
    stds = json.load(open("/Users/ayanbask/PycharmProjects/VAE/Autoencoder/class_stds_t1.json"))



    l1 = list(means.keys())
    l2 = list(stds.keys())

    # print('L1 ', l1)
    # print('L2 ', l2)

    mu_ = means[l1[0]]
    sigma_ = stds[l2[0]]

    arr = prior.gaussian(10, z_dim, mean=mu_, var=np.square(sigma_))

    for i in range(1, 10):
        mu = means[l1[i]]
        sigma = stds[l2[i]]
        # print(len(mu))
        # print(len(sigma))
        sample = prior.gaussian(10, z_dim, mean=mu, var=np.square(sigma))
        # print('Sample  ', sample)
        # print(type(sample))

        arr = np.concatenate((arr, sample), axis=0)
        # print('Shape  ', arr.shape)

    # arr = np.random.shuffle(arr)

    # print("GENERATING SAMPLE  ", arr)

    return arr



class VAE(object):
    model_name = "VAE"     # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size


        print("In Class ", self.dataset_name)
        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
            # parameters
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28
            self.n_classes = 10

            self.z_dim = z_dim         # dimension of noise-vector
            self.c_dim = 1

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5
            self.gamma = 5.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            self.data_X, self.data_y, self.data_X_val, self.data_y_val, self.data_X_te, self.data_y_te = load_mnist(self.dataset_name)
            # self.data_X, self.data_X_val = self.data_X_val, self.data_X
            # self.data_y, self.data_y_val = self.data_y_val, self.data_y

            self.data_X_dummy_1, self.data_y_dummy_1, self.data_X_dummy_1_val, self.data_y_dummy_1_val, self.data_X_dummy_1_te, self.data_y_dummy_1_te = load_e_mnist('emnist')
            # self.data_X_dummy_1, self.data_X_dummy_1_val = self.data_X_dummy_1_val, self.data_X_dummy_1
            # self.data_y_dummy_1, self.data_y_dummy_1_val = self.data_y_dummy_1_val, self.data_y_dummy_1

            print("Train Label  ", self.data_y_dummy_1.shape)
            print("Val Label", self.data_y_dummy_1_val.shape)
            print("Test Label", self.data_y_dummy_1_te.shape)


            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size
            self.num_batches_val = len(self.data_X_val) // self.batch_size

            print('Train Batches  ', self.num_batches)
            print('Val Batches  ', self.num_batches_val)

        # elif dataset_name == 'emnist' :
        #
        #     # parameters
        #     self.input_height = 28
        #     self.input_width = 28
        #     self.output_height = 28
        #     self.output_width = 28
        #     self.n_classes = 10
        #
        #     self.z_dim = z_dim         # dimension of noise-vector
        #     self.c_dim = 1
        #
        #     # train
        #     self.learning_rate = 0.0002
        #     self.beta1 = 0.5
        #
        #     # test
        #     self.sample_num = 64  # number of generated images to be saved
        #
        #     # load e-mnist and dummy mnist
        #     self.data_X, self.data_y = load_e_mnist(self.dataset_name)
        #     self.data_X_dummy_1, self.data_y_dummy_1 = load_mnist('mnist')
        #     # print("B  ", self.data_y_dummy_1.shape)
        #
        #     # get number of batches for a single epoch
        #     self.num_batches = len(self.data_X) // self.batch_size
        #     self.num_batches_val = len(self.data_X_val) // self.batch_size

        else:
            raise NotImplementedError

    # TODO: Change output returned from encoder layer from net to gaussian_params
    # Gaussian Encoder
    def encoder(self, x, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC62*4
        with tf.variable_scope("encoder", reuse=reuse):

            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='en_conv1'))
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='en_conv2'), is_training=is_training, scope='en_bn2'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(bn(linear(net, 1024, scope='en_fc3'), is_training=is_training, scope='en_bn3'))
            gaussian_params = linear(net, 2 * self.z_dim, scope='en_fc4')

            # The mean parameter is unconstrained
            mean = gaussian_params[:, :self.z_dim]
            # The standard deviation must be positive. Parameterize with a softplus and
            # add a small epsilon for numerical stability
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, self.z_dim:])

        return mean, stddev, gaussian_params

    # Bernoulli decoder
    def decoder(self, z, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("decoder", reuse=reuse) as scope:
            # out = np.zeros((self.batch_size, 14, 14, 64))
            try :
                net = tf.nn.relu(bn(linear(z, 1024, scope='defc1'), is_training=is_training, scope='de1'))
                net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='defc2'), is_training=is_training, scope='de2'))
                net = tf.reshape(net, [self.batch_size, 7, 7, 128])
                net = tf.nn.relu(
                    bn(deconv2d(net, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='dedc3'), is_training=is_training,
                       scope='de3'))

                out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='dedc4'))
            except ValueError :
                scope.reuse_variables()
                net = tf.nn.relu(bn(linear(z, 1024, scope='defc1'), is_training=is_training, scope='de1'))
                net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='defc2'), is_training=is_training, scope='de2'))
                net = tf.reshape(net, [self.batch_size, 7, 7, 128])
                net = tf.nn.relu(
                    bn(deconv2d(net, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='dedc3'), is_training=is_training,
                       scope='de3'))

                out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='dedc4'))

            return out


    # Classifier Network
    def classifier_ffn(self, h, is_training=True, reuse=False):

        with tf.variable_scope("classifier", reuse=reuse):
            h1 = tf.nn.relu(bn(linear(h, 512, scope='cl_fc1'), is_training=is_training, scope='cl_bn_fc1'))
            h2 = tf.nn.relu(bn(linear(h1, 256, scope='cl_fc2'), is_training=is_training, scope='cl_bn_fc2'))
            classifier_out = bn(linear(h2, 10, scope='cl_nn_fc3'), is_training=is_training, scope='cl_nn_bn_fc3')
            classifier_out_2t = bn(linear(classifier_out, 26, scope='cl_nn_fc4'), is_training=is_training, scope='cl_nn_bn_fc4')

        return classifier_out, classifier_out_2t


    def inference(self, task):
        print("abcde")

        print("Running Inference")
        bs = self.batch_size

        saved_model = "/Users/ayanbask/PycharmProjects/VAE/checkpoints/VAE_mnist_100_20/VAE/"
        ckpt = tf.train.latest_checkpoint(saved_model)
        print(ckpt)
        filename = ".".join([ckpt, 'meta'])
        print(filename)
        model_saver = tf.train.import_meta_graph(filename, clear_devices=True)
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)


        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            print('START EPOCH ', start_epoch)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            print("Start Epoch * num_batches ", self.num_batches)
            print('Start_Batch_ID ', start_batch_id)
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        preds = []
        means = []
        mnist = []
        stds = []
        test_loss = []
        test_loss_2t = []

        # with tf.Session() as sess:
        op = self.sess.graph.get_operations()
        print([m.values() for m in op][1])
        model_saver.restore(self.sess, ckpt)
        graph = tf.get_default_graph()
        inputs = graph.get_tensor_by_name("placeholders/real_images:0")
        # t1_labels = graph.get_tensor_by_name("placeholders/cl_y:0")

        z = tf.get_collection('z')[0]
        mu = tf.get_collection('mu')[0]
        sigma = tf.get_collection('sigma')[0]

        cl_loss = tf.get_collection('cl_loss')[0]
        cl_loss_2t = tf.get_collection('cl_loss_2t')[0]
        acc = tf.get_collection('acc')[0]
        acc_2t = tf.get_collection('acc_2t')[0]
        im = tf.get_collection('rec_image')

        # print('Start id  ', start_batch_id)
        def print_images(start_batch_id, batches, data, s) :
            for idx in range(start_batch_id, batches):

                if(task==1):
                    input_image_batch = data[idx*self.batch_size:(idx+1)*self.batch_size]
                else :
                    # input_image_batch = data[idx * self.batch_size:(idx + 1) * self.batch_size]
                    input_image_batch = self.data_X_dummy_1[idx*self.batch_size:(idx+1)*self.batch_size]

                batch_labels = self.data_y[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_labels_2t = self.data_y_dummy_1[idx*self.batch_size:(idx+1)*self.batch_size]
                mu_, sigma_,  _cl_loss, _cl_loss_2t, _acc, _acc_2t = self.sess.run([mu, sigma, cl_loss, cl_loss_2t, acc, acc_2t], feed_dict = {inputs:input_image_batch, self.cl_y: batch_labels, self.cl_y_2t: batch_labels_2t})



                reconstructed_images = self.sess.run(im, feed_dict = {inputs:input_image_batch})
                # print('Shape preds ', np.array(predictions).shape)
                # print('Shape recon ', np.array(reconstructed_images[0]).shape)
                # print('Shape recon ', np.array(reconstructed_images[1]).shape)

                # preds.extend(predictions)
                means.extend(mu_)
                stds.extend(sigma_)
                mnist.extend(batch_labels)

                s = 'mnist' if task==1 else 'e-mnist'
                t='mnist'


                if(s=='test') :
                    test_loss.extend(_cl_loss)
                    test_loss_2t.extend(_cl_loss_2t)

                tot_num_samples = min(self.sample_num, self.batch_size)
                manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                save_images(reconstructed_images[0][:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                            check_folder(
                                self.result_dir + '/' + self.model_dir) + '/' + 'reconstructed_' + s + '_after_task' + str(task) +'_' + s + '__{:04d}.png'.format(idx))


            # plt.title('Test Loss Curve')
            # plt.plot(test_loss)
            # plt.legend(loc='lower right')
            # # plt.plot([0, 1], [0, 1], 'r--')
            # # plt.xlim([0, 1])
            # # plt.ylim([0, 1])
            # plt.ylabel('------')
            # plt.xlabel('Test loss')
            # plt.savefig(check_folder(self.result_dir + '/' + self.model_dir) + '/' + 'loss_1t' + '.jpg')
            # plt.close()
            #


        print_images(start_batch_id, self.num_batches, self.data_X_dummy_1, "train")
        # print('Shape of predictions  ', np.array(preds).shape)
        # np.array(preds).dump(open('/Users/ayanbask/PycharmProjects/VAE/Autoencoder/auto_vectors_t1.npy', 'wb'))
        np.array(means).dump(open('/Users/ayanbask/PycharmProjects/VAE/Autoencoder/means_t' + str(task) + '.npy', 'wb'))
        np.array(stds).dump(open('/Users/ayanbask/PycharmProjects/VAE/Autoencoder/stds_t' + str(task) + '.npy', 'wb'))
        np.array(mnist).dump(open('/Users/ayanbask/PycharmProjects/VAE/Autoencoder/labels_t' + str(task) + '.npy', 'wb'))

        print('means shape ', np.array(means).shape)
        print('stds shape ', np.array(stds).shape)
        print('label shape ', np.array(mnist).shape)


        print("*****************VALIDATION LOSSES IN INFERENCE***********************")
        print(self.valid_loss_t1)
        np.array(self.valid_loss_t1).dump(open('/Users/ayanbask/PycharmProjects/VAE/Autoencoder/' + str(task) + 'TASK_valid_loss_t' + str(task) + '.npy', 'wb'))
        print(np.array(self.valid_loss_t1).shape)
        print(self.valid_loss_t2)
        np.array(self.valid_loss_t2).dump(open('/Users/ayanbask/PycharmProjects/VAE/Autoencoder/' + str(task) + 'TASK_valid_loss_t' + str(task) + '.npy', 'wb'))

        print(np.array(self.valid_loss_t2).shape)

        g1=[]
        g2=[]

        for i in list(self.valid_loss_t1) :
            s1 = sum(i)/len(i)
            g1.append(s1)

        for j in list(self.valid_loss_t2) :
            s2 = sum(j)/len(j)
            g2.append(s2)

        plt.title('Validation Loss Curve')
        plt.plot(g1)
        plt.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1], 'r--')
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        plt.ylabel('Validation loss')
        plt.xlabel('Epochs')
        plt.savefig(check_folder(self.result_dir + '/' + self.model_dir) + '/' + str(task) + 'TASK_loss_1t' + '.jpg')
        plt.close()

        plt.title('Validation Loss Curve')
        plt.plot(g2)
        plt.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1], 'r--')
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        plt.ylabel('Validation loss')
        plt.xlabel('Epochs')
        plt.savefig(check_folder(self.result_dir + '/' + self.model_dir) + '/' + str(task) + 'TASK_loss_2t' + '.jpg')
        plt.close()



        preds=[]
        means=[]
        stds=[]
        mnist=[]

        print_images(start_batch_id, self.num_batches_val, self.data_X_dummy_1_val, "val")
        # preds = []


        t1_after_t2 = []
        g3 = []
        g4 = []

        for idx in range(start_batch_id, self.num_batches):
            input_image_batch = self.data_X[idx * self.batch_size:(idx + 1) * self.batch_size]
            # input_image_batch = self.data_X_dummy_1[idx * self.batch_size:(idx + 1) * self.batch_size]

            batch_labels = self.data_y[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_labels_2t = self.data_y_dummy_1[idx * self.batch_size:(idx + 1) * self.batch_size]
            _acc, _acc_2t = self.sess.run([acc, acc_2t], feed_dict={inputs: input_image_batch, self.cl_y: batch_labels, self.cl_y_2t: batch_labels_2t})

            t1_after_t2.append(_acc)
            g4.append(_acc_2t)

        print("TESTINGGGGGGGGG  ", type(t1_after_t2))
        print("TESTINGGGGGGGGG  ", t1_after_t2)

        # exit()
        # 
        # for i in list(t1_after_t2) :
        #     s3 = sum(i)/len(i)
        #     g3.append(s3)



        plt.title('MNIST Accuracy after Task 2')
        plt.plot(t1_after_t2)
        plt.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1], 'r--')
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        plt.ylabel('MNIST Accuracy')
        plt.xlabel('Epochs')
        plt.savefig(check_folder(self.result_dir + '/' + self.model_dir) + '/' + 'MNIST' + '_accuracy_after_task_2' + '.jpg')
        plt.close()




        print(g1)
        print(g2)

        return preds



    def square(self, sigma):
        return float(sigma*sigma)



    def build_model(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        with tf.name_scope("placeholders"):
            # images
            self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

            # noises
            self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')


            # # labels mnist
            # self.cl_y = tf.placeholder(tf.float32, [bs, 10], name='cl_y')
            # labels e-mnist
            self.cl_y = tf.placeholder(tf.float32, [bs, 10], name='cl_y')
            self.cl_y_2t = tf.placeholder(tf.float32, [bs, 26], name='cl_y_2t')



        # encoding
        self.mu, sigma, h = self.encoder(self.inputs, is_training=True, reuse=False)

        tf.add_to_collection("mu", self.mu)
        tf.add_to_collection("sigma", sigma)

        self.sigma = sigma

        # sampling by re-parameterization technique
        z = self.mu + self.sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)
        tf.add_to_collection("z", z)

        # decoding
        out = self.decoder(z, is_training=True, reuse=False)
        self.out = tf.clip_by_value(out, 1e-8, 1 - 1e-8)


        tf.add_to_collection("rec_image", self.out)




        """" Testing """
        # for test
        self.fake_images = self.decoder(self.z, is_training=False, reuse=True)
        # self.fake_images = self.decoder(z, is_training=True, reuse=False)

        tf.add_to_collection("fake_images", self.fake_images)


        # loss
        marginal_likelihood = tf.reduce_sum(self.inputs * tf.log(self.out) + (1 - self.inputs) * tf.log(1 - self.out),
                                            [1, 2])
        self.neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)

        KL_divergence = 0.5 * tf.reduce_sum(tf.square(self.mu) + tf.square(self.sigma) - tf.log(1e-8 + tf.square(self.sigma)) - 1, [1])

        self.KL_divergence = tf.reduce_mean(KL_divergence)





        # self.wass_loss = sliced_wasserstein_tf(self.inputs, self.fake_images, dir_repeats=dir_repeats, dirs_per_repeat=dirs_per_repeat)
        self.wass_loss = sliced_wasserstein_tf(self.out, self.fake_images, dir_repeats=dir_repeats, dirs_per_repeat=dirs_per_repeat)
        tf.add_to_collection("wass_loss", self.wass_loss)

        # ELBO = -self.neg_loglikelihood - self.KL_divergence - self.wass_loss
        ELBO = -self.neg_loglikelihood - (self.gamma * self.KL_divergence)
        t2_loss = -self.neg_loglikelihood - self.wass_loss
        tf.add_to_collection("t2_loss", t2_loss)

        self.loss = -ELBO
        self.t2_loss = -t2_loss


        classifier_out, classifier_out_2t = self.classifier_ffn(z)

        self.classifier_out = tf.nn.softmax(classifier_out)
        self.classifier_out_2t = tf.nn.softmax(classifier_out_2t)
        tf.add_to_collection("classifier_out", self.classifier_out)
        tf.add_to_collection("classifier_out_2t", self.classifier_out_2t)


        # Define the loss function, optimizer, and accuracy
        self.cl_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.cl_y, logits=self.classifier_out), name='loss')
        self.cl_loss_2t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.cl_y_2t, logits=self.classifier_out_2t), name='loss')
        tf.add_to_collection("cl_loss", self.cl_loss)
        tf.add_to_collection("cl_loss_2t", self.cl_loss_2t)

        self.final_classifier_loss = self.cl_loss + self.cl_loss_2t
        self.cl_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='Adam-op').minimize(self.final_classifier_loss)
        tf.add_to_collection("final_classifier_loss", self.final_classifier_loss)


        self.correct_prediction = tf.equal(tf.argmax(self.classifier_out, 1), tf.argmax(self.cl_y, 1), name='correct_pred')
        self.correct_prediction_2t = tf.equal(tf.argmax(self.classifier_out_2t, 1), tf.argmax(self.cl_y_2t, 1), name='correct_pred')
        self.acc = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name='classifier_accuracy')
        self.acc_2t = tf.reduce_mean(tf.cast(self.correct_prediction_2t, tf.float32), name='classifier_accuracy_2t')
        tf.add_to_collection("acc", self.acc)
        tf.add_to_collection("acc_2t", self.acc_2t)



        """ Training """
        # optimizers
        t_vars = tf.trainable_variables()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                      .minimize(self.loss, var_list=t_vars)
            # self.optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
            #           .minimize(self.t2_loss, var_list=t_vars)
            self.optim_t2 = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                      .minimize(self.t2_loss, var_list=t_vars)


        """ Summary """
        nll_sum = tf.summary.scalar("nll", self.neg_loglikelihood)
        kl_sum = tf.summary.scalar("kl", self.KL_divergence)
        wass_loss = tf.summary.scalar("wass", self.wass_loss)

        loss_sum = tf.summary.scalar("t1_loss (neg likelihood + KL)", self.loss)
        t2_loss_sum = tf.summary.scalar("t2_loss (neg likelihood + Wassertein)", self.t2_loss)

        classifier_accuracy = tf.summary.scalar("cl_accuracy_mnist", self.acc)
        classifier_accuracy_2t = tf.summary.scalar("cl_accuracy_e-mnist", self.acc_2t)

        # final summary operations
        self.merged_summary_op = tf.summary.merge_all()

    def train(self, task):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training Autoencoder

        # flag=0
        # try :
        #
        #     vectors = np.load('/Users/ayanbask/PycharmProjects/VAE/Autoencoder/auto_vectors_t1.npy', allow_pickle=True)
        #     mu = np.mean(vectors)
        #     sigma = np.std(vectors)
        #     self.sample_z = prior.gaussian(self.batch_size, self.z_dim, mean=mu, var=self.square(sigma))
        # except FileNotFoundError :
        #     flag=1
        #
        #     mu = 0
        #     sigma = 1
        #     self.sample_z = prior.gaussian(self.batch_size, self.z_dim, mean=0, var=1)
        #
        # if(flag==0) :
        #     print("SAMPLING FROM LEARNT TASK t-1 GAUSSIAN.....in train")
        # else :
        #     print("SAMPLING FROM RANDOM GAUSSIAN.....in train")


        if(task==1) :
            self.sample_z = prior.gaussian(self.batch_size, self.z_dim, mean=0, var=1)
            batch_z = prior.gaussian(self.batch_size, self.z_dim, mean=0, var=1)
        else:
            print("Entered T1 Sampler")
            self.sample_z = generate_stratified_samples(self.batch_size, self.z_dim)
            batch_z = generate_stratified_samples(self.batch_size, self.z_dim)

        # print('BATCH z ', batch_z)
        # print('Task ', task)

        # with tf.control_dependencies([z_shape]) :
            # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name + '/' + 'task_' + str(task), self.sess.graph)
        self.writer.add_graph(self.sess.graph)

        self.valid_writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name + '/' + 'valid_task_' + str(task), self.sess.graph)
        self.valid_writer.add_graph(self.sess.graph)

        self.valid_mnist_writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name + '/' + 'mnist_valid_after_task_' + str(task), self.sess.graph)
        self.valid_mnist_writer.add_graph(self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        print('CHECKPOINT COUNTER  ', checkpoint_counter )
        if could_load:
            # start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_epoch = int(checkpoint_counter)
            print("Start Epoch ", start_epoch)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            print("Start Epoch * num_batches ", self.num_batches)
            print('Start Batch ID ', start_batch_id)
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        print("START EPOCH  ", start_epoch)

        # loop for epoch
        start_time = time.time()

        self.valid_loss_t1 = []
        self.valid_loss_t2 = []

        print(np.array(self.data_X).shape)
        print(np.array(self.data_y).shape)

        print("*"*60)
        print(start_epoch, start_batch_id, self.num_batches, self.num_batches_val)


        for epoch in range(start_epoch, self.epoch):

            vt1 = []
            vt2 = []
            # get batch data
            # btchs = self.num_batches if task==1 else self.num_batches_val

            for idx in range(0, self.num_batches):
            # for idx in range(0, btchs):

                if(task==1):
                    batch_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                else :
                    batch_images = self.data_X_dummy_1[idx*self.batch_size:(idx+1)*self.batch_size]

                batch_labels = self.data_y[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_labels_2t = self.data_y_dummy_1[idx*self.batch_size:(idx+1)*self.batch_size]
                # print('ERROR SHAPE ', batch_labels_2t.shape)


                # batch_z = prior.gaussian(self.batch_size, self.z_dim)
                # batch_z = self.mu + self.sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)
                # dummy_batch_z = tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)
                #
                # # Get mu and sigma
                # mu, sigma = self.sess.run([self.mu, self.sigma], feed_dict={self.inputs: batch_images})
                # print('mu  ', mu)
                # print('sigma  ', sigma)
                # batch_z = prior.gaussian(self.batch_size, self.z_dim, mean=mu, var=np.square(sigma))

                # batch_z = generate_stratified_samples(self.batch_size, self.z_dim)


                # update autoencoder
                _, summary_str, loss, t2_loss, nll_loss, kl_loss, wass_loss = self.sess.run([self.optim_t2, self.merged_summary_op, self.loss, self.t2_loss, self.neg_loglikelihood, self.KL_divergence, self.wass_loss],
                                               feed_dict={self.inputs: batch_images, self.z: batch_z, self.cl_y: batch_labels, self.cl_y_2t: batch_labels_2t})
                self.writer.add_summary(summary_str, counter)



                # update classifier
                _, summary_str_cl, final_cl_loss, cl_loss, cl_loss_2t, cl_accuracy, cl_accuracy_2t = self.sess.run([self.cl_optimizer, self.merged_summary_op, self.final_classifier_loss, self.cl_loss, self.cl_loss_2t, self.acc, self.acc_2t],
                                               feed_dict={self.inputs: batch_images, self.z: batch_z, self.cl_y: batch_labels, self.cl_y_2t: batch_labels_2t})
                self.writer.add_summary(summary_str_cl, counter)
                self.writer.flush()


                if(task==1) :
                    print("TRAIN CL_ACC : ", cl_accuracy)
                else :
                    print("TRAIN CL_ACC : ", cl_accuracy_2t)
                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, t2_loss: %.8f, nll: %.8f, kl: %.8f, final_cl_loss: %.8f, cl_loss: %.8f, cl_loss_2t: %.8f, wass_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, loss, t2_loss, nll_loss, kl_loss, final_cl_loss, cl_loss, cl_loss_2t, wass_loss))



            if (task == 2):
                for idx in range(0, self.num_batches_val) :

                    batch_images_mnist_val = self.data_X_val[idx * self.batch_size:(idx + 1) * self.batch_size]
                    batch_labels_mnist_val = self.data_y_val[idx * self.batch_size:(idx + 1) * self.batch_size]
                    batch_labels_2t_mnist_val = self.data_y_dummy_1_val[idx * self.batch_size:(idx + 1) * self.batch_size]

                    summary_str_mnist_eval, loss_mnist_eval, t2_loss_mnist_eval, nll_loss_mnist_eval, kl_loss_mnist_eval, wass_loss_mnist_eval, \
                    summary_str_cl_mnist_eval, final_cl_loss_mnist_eval, cl_loss_mnist_eval, cl_loss_2t_mnist_eval, cl_accuracy_mnist_eval, \
                    cl_accuracy_2t_mnist_eval = self.sess.run([self.merged_summary_op, self.loss, self.t2_loss, self.neg_loglikelihood,
                                                self.KL_divergence, self.wass_loss,
                                                self.merged_summary_op, self.final_classifier_loss,
                                                self.cl_loss, self.cl_loss_2t, self.acc, self.acc_2t],
                                                feed_dict={self.inputs: batch_images_mnist_val, self.z: batch_z, self.cl_y: batch_labels_mnist_val,
                                                 self.cl_y_2t: batch_labels_2t_mnist_val})

                    self.valid_mnist_writer.add_summary(summary_str_mnist_eval, counter)
                    self.valid_mnist_writer.add_summary(summary_str_cl_mnist_eval, counter)





                """
                # save training Autoencoder for every 300 steps
                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.z: self.sample_z})

                    # samples = self.fake_images
                    # samples = self.out

                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_task1_{:02d}_{:04d}.png'.format(
                                    epoch, idx))

                """


            # get batch data

            for idx in range(0, self.num_batches_val):
                # print(idx)
                if(task==1) :
                    batch_images_val = self.data_X_val[idx*self.batch_size:(idx+1)*self.batch_size]
                else :
                    batch_images_val = self.data_X_dummy_1_val[idx*self.batch_size:(idx+1)*self.batch_size]
                    batch_images_val_mnist_eval = self.data_X_val[idx * self.batch_size:(idx + 1) * self.batch_size]

                batch_labels_val = self.data_y_val[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_labels_2t_val = self.data_y_dummy_1_val[idx*self.batch_size:(idx+1)*self.batch_size]



                # autoencoder + classifier
                summary_str, loss, t2_loss, nll_loss, kl_loss, wass_loss, summary_str_cl, final_cl_loss, cl_loss, cl_loss_2t, \
                cl_accuracy, cl_accuracy_2t = self.sess.run([self.merged_summary_op, self.loss, self.t2_loss, self.neg_loglikelihood,
                                self.KL_divergence, self.wass_loss,
                                self.merged_summary_op, self.final_classifier_loss,
                                self.cl_loss, self.cl_loss_2t, self.acc, self.acc_2t],
                                feed_dict={self.inputs: batch_images_val, self.z: batch_z, self.cl_y: batch_labels_val, self.cl_y_2t: batch_labels_2t_val})
                self.valid_writer.add_summary(summary_str, counter)





                # # classifier
                #  = self.sess.run([],
                #                                feed_dict={self.inputs: batch_images_val, self.z: batch_z, self.cl_y: batch_labels_val, self.cl_y_2t: batch_labels_2t_val})


                if(task==1) :
                    print("TEST_CL_ACC", cl_accuracy)
                else :
                    print("TEST_CL_ACC", cl_accuracy_2t)


                self.valid_writer.add_summary(summary_str_cl, counter)
                self.valid_writer.flush()

                vt1.append(cl_loss)
                vt2.append(cl_loss_2t)

                # display validation status
                counter += 1
                print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<On VALIDATION SET>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, t2_loss: %.8f, nll: %.8f, kl: %.8f, final_cl_loss: %.8f, cl_loss: %.8f, cl_loss_2t: %.8f, wass_loss: %.8f" \
                      % (epoch, idx, self.num_batches_val, time.time() - start_time, loss, t2_loss, nll_loss, kl_loss, final_cl_loss, cl_loss, cl_loss_2t, wass_loss))

            self.valid_loss_t1.append(vt1)
            self.valid_loss_t2.append(vt2)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            # self.save(self.checkpoint_dir, counter)
            self.save(self.checkpoint_dir, epoch)

            # show temporal Autoencoder
            self.visualize_results(epoch, task)
            # self.visualize_results()

        print("###############VALIDATION LOSSES##################")
        print(self.valid_loss_t1)
        print(np.array(self.valid_loss_t1).shape)
        print(self.valid_loss_t2)
        print(np.array(self.valid_loss_t2).shape)

        # save model for final step
        # self.save(self.checkpoint_dir, counter)
        self.save(self.checkpoint_dir, self.epoch)



    def compute_means_stds(self, task) :
        class_means = {}
        class_stds = {}


        means = np.load('/Users/ayanbask/PycharmProjects/VAE/Autoencoder/means_t' + str(task) + '.npy', allow_pickle=True)
        stds = np.load('/Users/ayanbask/PycharmProjects/VAE/Autoencoder/stds_t' + str(task) + '.npy', allow_pickle=True)
        labels = np.load('/Users/ayanbask/PycharmProjects/VAE/Autoencoder/labels_t' + str(task) + '.npy', allow_pickle=True)
        lab = [-1] * len(labels)

        print(means.shape)
        print(stds.shape)
        print(labels.shape)
        # print(labels[0])
        for i, l in enumerate(labels):
            for j in range(0, len(l)):
                if (l[j] == 1):
                    lab[i] = j

        # print(labels[1], lab[1])

        df = pd.DataFrame()

        df["means"] = list(means)
        df["stds"] = list(stds)
        df["labels"] = list(lab)

        grouped = df.groupby('labels').groups

        for group in grouped:
            means_sum = 0
            stds_sum = 0
            idx = grouped[group].tolist()
            for i in idx:
                means_sum += means[i]
                stds_sum += stds[i]

            class_means[group] = (means_sum / len(idx))
            class_stds[group] = stds_sum / len(idx)

        for g in class_means:
            class_means[g] = class_means[g].tolist()

        for s in class_stds:
            class_stds[s] = class_stds[s].tolist()

        print(class_means)
        print(class_stds)

        json.dump(dict(class_means), open("/Users/ayanbask/PycharmProjects/VAE/Autoencoder/class_means_t" + str(task) + ".json", 'w'))
        json.dump(dict(class_stds), open("/Users/ayanbask/PycharmProjects/VAE/Autoencoder/class_stds_t" + str(task) + ".json", 'w'))




    def visualize_results(self, epoch, task):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        # flag=0
        # try :
        #     vectors = np.load('/Users/ayanbask/PycharmProjects/VAE/Autoencoder/auto_vectors_t1.npy', allow_pickle=True)
        #     mu = np.mean(vectors)
        #     sigma = np.std(vectors)
        #     z_sample = prior.gaussian(self.batch_size, self.z_dim, mean=mu, var=self.square(sigma))
        # except FileNotFoundError :
        #     flag=1
        #     mu = 0
        #     sigma = 1
        #
        #     z_sample = prior.gaussian(self.batch_size, self.z_dim, mean=0, var=1)


        # if flag==0 :
        #     print("SAMPLING FROM LEARNT t-1 GAUSSIAN.....")
        # else :
        #     print("SAMPLING FROM RANDOM GAUSSIAN.....")

        # z_sample = prior.gaussian(self.batch_size, self.z_dim, mean=mu, var=self.square(sigma))


        if(task==1) :
            z_sample = prior.gaussian(self.batch_size, self.z_dim, mean=0, var=1)
        else :
            z_sample = generate_stratified_samples(self.batch_size, self.z_dim)


        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(
                        self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes_task_1.png')

        """ learned manifold """
        if self.z_dim == 2:
            assert self.z_dim == 2

            z_tot = None
            id_tot = None
            for idx in range(0, 100):
                # randomly sampling
                id = np.random.randint(0, self.num_batches)
                batch_images = self.data_X[id * self.batch_size:(id + 1) * self.batch_size]
                batch_labels = self.data_y[id * self.batch_size:(id + 1) * self.batch_size]

                z = self.sess.run(self.mu, feed_dict={self.inputs: batch_images})

                if idx == 0:
                    z_tot = z
                    id_tot = batch_labels
                else:
                    z_tot = np.concatenate((z_tot, z), axis=0)
                    id_tot = np.concatenate((id_tot, batch_labels), axis=0)

            save_scattered_image(z_tot, id_tot, -4, 4, name=check_folder(
                self.result_dir + '/' + self.model_dir + '/' + self.model_name) + '_epoch%03d' % epoch + '_learned_manifold_after_task_1.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        # print('CKPT Dir ', checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # print(os.path.join(checkpoint_dir, self.model_name+'.model'))
        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        # checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)
        checkpoint_dir = "/Users/ayanbask/PycharmProjects/VAE/checkpoints/VAE_mnist_100_20/VAE/"
        print("Checkpoint Dir ", checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))

            print("CHECKPOINT COUNTER  ", counter)
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            print("CHECKPOINT COUNTER  ", 0)
            return False, 0
