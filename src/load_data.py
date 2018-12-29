from __future__ import absolute_import
from keras.datasets.cifar import load_batch
from keras.utils.data_utils import get_file
import keras
from keras import backend as K
import numpy as np
from scipy.io import loadmat


def X_data_reshape(X, image_dim):
    X = X.reshape(X.shape[0], 3, image_dim, image_dim)
    if K.image_data_format() == 'channels_last':
        X = X.transpose(0, 2, 3, 1)

    return X


def load_celeba_aligned(image_dim, ith):
    """
    # Load aligned images of CelebA dataset - all.
    # Returns Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    # datafile = "/data/jihyec/data/celeba/img_align_celeba_dim%d_%d.mat" % (image_dim, ith)
    datafile = "/home/jihyec/data/celeba/img_align_celeba-mtcnnpy_dim%d_mat/img_align_celeba-mtcnnpy_dim%d_%d.mat" % (image_dim, image_dim, ith)
    # note that data are already shuffled

    """
    data = loadmat(datafile)
    y_train = data['y_train']
    y_test = data['y_test']
    x_train = data['x_train']
    x_test = data['x_test']
    """


    import h5py
    with h5py.File(datafile, 'r') as file:
        x_train = np.array(file['x_train'])
        x_test = np.array(file['x_test'])
        y_train = np.array(file['y_train'])
        y_test = np.array(file['y_test'])
    x_train = x_train.transpose(1, 0)
    x_test = x_test.transpose(1, 0)
    y_train = y_train.transpose(1, 0)
    y_test = y_test.transpose(1, 0)

    x_train = X_data_reshape(x_train, image_dim)
    x_test = X_data_reshape(x_test, image_dim)
    """
    print('shape of one x_train')
    print(np.shape(x_train))
    print(np.shape(y_train))
    print('shape of one x_test')
    print(np.shape(x_test))
    print(np.shape(y_test))
    """
    return (x_train, y_train), (x_test, y_test)


def load_data_celeba_multiple(image_dim, start_set, end_set):
    """
    Preprocess CelebA dataset
    :return: NumPy arrays
    """
    img_rows = image_dim  # 32
    img_cols = image_dim  # 32
    # nb_classes = 20  # 10

    (X_train, y_train), (X_test, y_test) = load_celeba_aligned(image_dim, start_set)
    for ith_subset in range(start_set+1, end_set+1):
        (ith_X_train, ith_y_train), (ith_X_test, ith_y_test) = load_celeba_aligned(image_dim, ith_subset)
        X_train = np.vstack((X_train, ith_X_train))
        X_test = np.vstack((X_test, ith_X_test))
        y_train = np.vstack((y_train, ith_y_train))
        y_test = np.vstack((y_test, ith_y_test))


    if keras.backend.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)

    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
 
    # Change pixel values to 0 to 1 values
    min_pixel = min(np.min(X_train), np.min(X_test))
    max_pixel = max(np.max(X_train), np.max(X_test))
    # faces1 = (faces1 - min_pixel) / (max_pixel - min_pixel)
    # faces2 = (faces2 - min_pixel) / (max_pixel - min_pixel)

    # print(min_pixel)
    # print(max_pixel)

    print(np.shape(X_train))
    print(np.shape(y_train))
    print(np.shape(X_test))
    print(np.shape(y_test))
 
    return (X_train, y_train), (X_test, y_test)


def load_vgg_aligned_easy_set(imdim):
    """
    Load aligned images of VGG dataset - easy set of 20 subjects.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.



    dirname = '/data/jihyec/data'  # 'cifar-10-batches-py'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file(dirname, origin=origin, untar=True)

    num_train_samples = 500 * 20 * 0.8  # 50000

    x_train = np.empty((num_train_samples, 3, 80, 80), dtype='uint8')    # 32, 32
    y_train = np.empty((num_train_samples,), dtype='uint8')


    for i in range(1, 8):  # 1,6
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 1000: i * 1000, :, :, :],   # 10000
         y_train[(i - 1) * 1000: i * 1000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)
    """

    # load the images aligned and  tightly cropped for VGGNet
    """
    data = loadmat("/data/jihyec/data/vgg/vgg_easy_set_dim%d_cleaned.mat" % imdim) # for_VGGNet
    y_train = data['y_train']
    y_test = data['y_test']
    x_train = data['x_train']
    x_test = data['x_test']

    """
    import h5py
    with h5py.File("/data/jihyec/data/vgg/vgg_easy_set-mtcnnpy_dim%d.mat" % imdim, 'r') as file:
        x_train = np.array(file['x_train'])
        x_test = np.array(file['x_test'])
        y_train = np.array(file['y_train'])
        y_test = np.array(file['y_test'])
    x_train = x_train.transpose(1, 0)
    x_test = x_test.transpose(1, 0)


    x_train = X_data_reshape(x_train, imdim)
    x_test = X_data_reshape(x_test, imdim)
    y_train = y_train - 1  # range of labels from 1~20 to 0~19
    y_test = y_test - 1

    print(y_train.shape)  # len(y_train)
    print(len(y_test))
    #y_train = np.reshape(y_train, (y_train.shape[1], 1))
    #y_test = np.reshape(y_test, (y_test.shape[1], 1))

    return (x_train, y_train), (x_test, y_test)


def load_lfw_aligned(image_dim, ith_fold, view):
    if view == 1:
        datafile = "/data/jihyec/data/lfw/lfw-deepfunneled_dim%d_view%d.mat" % (image_dim, view)
        data = loadmat(datafile)
        X_train1 = data['train_images1']
        X_train2 = data['train_images2']
        X_test1 = data['test_images1']
        X_test2 = data['test_images2']

        X_train1 = X_data_reshape(X_train1, image_dim)
        X_train2 = X_data_reshape(X_train2, image_dim)
        X_test1 = X_data_reshape(X_test1, image_dim)
        X_test2 = X_data_reshape(X_test2, image_dim)

        return X_train1, X_train2, X_test1, X_test2

    else:
        datafile = "/data/jihyec/data/lfw/lfw-deepfunneled_dim%d_view%d_%d.mat" % (image_dim, view, ith_fold)
        data = loadmat(datafile)
        X_eval1 = data['eval_images1']
        X_eval2 = data['eval_images2']

        X_eval1 = X_data_reshape(X_eval1, image_dim)
        X_eval2 = X_data_reshape(X_eval2, image_dim)

        return X_eval1, X_eval2

