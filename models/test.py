import tensorflow as tf
import numpy as np
import time
import os
import gzip




dir_repeats = 4
dirs_per_repeat = 128

def sliced_wasserstein(A, B, dir_repeats, dirs_per_repeat):

    assert A.shape == B.shape
    print(A.shape, B.shape)                                             # (neighborhood, descriptor_component)
    A = tf.reshape(A, shape = [-1, 28, 28])
    B = tf.reshape(B, shape = [-1, 28, 28])
    assert A.shape == B.shape
    print(A.shape, B.shape)
    exit()

    results = []
    for repeat in range(dir_repeats):
        dirs = np.random.randn(A.shape[1], dirs_per_repeat)             # (descriptor_component, direction)
        dirs /= np.sqrt(np.sum(np.square(dirs), axis=0, keepdims=True)) # normalize descriptor components for each direction
        dirs = dirs.astype(np.float32)
        projA = np.matmul(A, dirs)                                      # (neighborhood, direction)
        projB = np.matmul(B, dirs)
        projA = np.sort(projA, axis=0)                                  # sort neighborhood projections for each direction
        projB = np.sort(projB, axis=0)
        dists = np.abs(projA - projB)                                   # pointwise wasserstein distances
        results.append(np.mean(dists))                                  # average over neighborhoods and directions

    return tf.reduce_sum(np.mean(results, dtype=np.float32))




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

        # with tf.variable_scope('v1') as scope:
        #     try:
        #         v = tf.get_variable(var, shape=[A.shape[1], dirs_per_repeat])
        #     except ValueError:
        #         scope.reuse_variables()
        #         v = tf.get_variable(var)


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


    return tf.reduce_mean(results)



def load_mnist(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data('/Users/ayanbask/PycharmProjects/VAE/data/mnist/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data('/Users/ayanbask/PycharmProjects/VAE/data/mnist/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data('/Users/ayanbask/PycharmProjects/VAE/data/mnist/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data('/Users/ayanbask/PycharmProjects/VAE/data/mnist/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec



def build_model():
    # parameters

    image_dims = [28, 28, 1]
    bs = 32
    input_height = 28
    input_width = 28
    output_height = 28
    output_width = 28
    n_classes = 10

    c_dim = 1

    # train
    learning_rate = 0.0002
    beta1 = 0.5

    # test
    sample_num = 64  # number of generated images to be saved

    # load mnist
    data_X, data_y = load_mnist('mnist')
    print(data_X.shape)
    print(data_y.shape)

    data_a = data_X[:30000,:,:,:]
    data_a_lab = data_y[:30000,:]
    data_b = data_X[30000:60000,:,:,:]
    data_b_lab = data_y[30000:60000, :]

    print(data_a.shape)
    print(data_a_lab.shape)
    print(data_b.shape)
    print(data_b_lab.shape)
    # exit()

    # get number of batches for a single epoch
    num_batches = len(data_X) // bs


    # print(data_X[0,:,:].shape)
    # print(data_X[0,:,:].reshape(28,28))
    # exit()

    # print(sliced_wasserstein(data_X[0,:,:].reshape(28,28), data_X[1,:,:].reshape(28,28), dir_repeats, dirs_per_repeat))
    # exit()


    # some parameters


    """ Graph Input """
    # images
    a = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

    # noises
    b = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')




    # a_hat = tf.compat.v1.py_func(sliced_wasserstein, [a, b], tf.float32)
    a_hat = sliced_wasserstein_tf(a,b, dir_repeats=dir_repeats, dirs_per_repeat=dirs_per_repeat)
    # b_hat = tf.compat.v1.py_func(my_func, [input], tf.float32)

    print_a_hat = tf.print("a_hat", a_hat)
    print_a_hat_shape = tf.print("a_hat", tf.shape(a_hat))

    # wass = sliced_wasserstein(a, b, dir_repeats, dirs_per_repeat)
    # dummy = tf.Variable(0, dtype=float)
    # wass = dummy.assign(a_hat)
    # print(tf.shape(wass))
    # print_wass = tf.print("WASS metric", wass)
    # print(wass)


    """ Training """
    # optimizers
    with tf.control_dependencies([print_a_hat, print_a_hat_shape]):
        t_vars = tf.trainable_variables()
        # print(t_vars)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        with tf.control_dependencies([print_a_hat_shape, print_a_hat]):
            optim = tf.train.AdamOptimizer(learning_rate * 5, beta1=beta1) \
                .minimize(a_hat, var_list=t_vars)

    loss_sum = tf.summary.scalar("wass loss", a_hat)

    merged_summary_op = tf.summary.merge_all()

    # initialize all variables
    init = tf.global_variables_initializer()

    # tf.reset_default_graph()

    with tf.Session() as sess :

        sess.run(init)
        # saver to save model
        saver = tf.train.Saver()

        # summary writer
        writer = tf.summary.FileWriter("testing", sess.graph)

        writer.add_graph(sess.graph)
        # loop for epoch
        start_time = time.time()

        counter = 1
        for epoch in range(0, 5):

            # get batch data
            print('Num batches', num_batches)
            for idx in range(0, num_batches):
                batch_images1 = data_a[idx * bs:(idx + 1) * bs]
                batch1_labels1 = data_a_lab[idx * bs:(idx + 1) * bs]
                batch_images2 = data_b[idx * bs:(idx + 1) * bs]
                batch_labels2 = data_b_lab[idx * bs:(idx + 1) * bs]

                # batch_z = self.mu + self.sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)

                # update autoencoder
                _, summary_str, loss = sess.run(
                    [optim, merged_summary_op, a_hat],
                    feed_dict={a: batch_images1, b: batch_images2,})


                writer.add_summary(summary_str, counter)


                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f" \
                      % (epoch, idx, num_batches, time.time() - start_time, loss))




build_model()

