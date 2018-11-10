from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from keras import backend as K

import numpy as np
import random
import scipy.io


def binary_crossentropy_custom(y_true, y_pred):
    # y_pred : N by 40 tensor
    # K.binary_crossentropy(y_true, y_pred): N by 40
    # the original binary_crossentropy returns K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

    bin_cross = K.binary_crossentropy(y_true[:, 0], y_pred[:, 0])
    for i in range(1, 40):  # 1~39
        ith_bin_cross = K.binary_crossentropy(y_true[:, i], y_pred[:, i])
        bin_cross = K.concatenate((bin_cross, ith_bin_cross), axis=-1)

    return K.mean(bin_cross, axis=-1)
    #return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


def shuffle_dataset(X, y):
    # Shuffle the dataset
    tmp = np.array(np.arange(len(y)))
    random.shuffle(tmp)
    X = X[tmp]
    y = y[tmp]

    return X, y


def balance_dataset(X_train, y_train, top_unbal_attr_idx):
    input_dim = np.shape(X_train)[1]
    X_train_balanced = np.empty((0, input_dim, input_dim, 3), float)
    y_train_balanced = np.empty((0, 40), float)

    for i in range(0, len(top_unbal_attr_idx)):
        idx0 = np.array(np.where(y_train[:, top_unbal_attr_idx[i]] == -1))
        idx0 = idx0[0]
        l0 = len(idx0)
        idx1 = np.array(np.where(y_train[:, top_unbal_attr_idx[i]] == 1))
        idx1 = idx1[0]
        l1 = len(idx1)

        if l0 > l1:
            tmp = np.array(np.arange(l0))
            random.shuffle(tmp)
            tmp = tmp[:l1]
            X, y = shuffle_dataset(X=np.concatenate((X_train[idx0[tmp]], X_train[idx1]), axis=0),
                                   y=np.append(y_train[idx0[tmp]], y_train[idx1], axis=0))
        else:
            tmp = np.array(np.arange(l1))
            random.shuffle(tmp)
            tmp = tmp[:l0]
            X, y = shuffle_dataset(X=np.concatenate((X_train[idx1[tmp]], X_train[idx0]), axis=0),
                                   y=np.append(y_train[idx1[tmp]], y_train[idx0], axis=0))

        y_train_balanced = np.append(y_train_balanced, y, axis=0)
        X_train_balanced = np.concatenate((X_train_balanced, X), axis=0)

        del X, y

    return X_train_balanced, y_train_balanced


def convert_y_range(y_train, y_test):
    idx_train = np.array(np.where(y_train == -1))
    y_train[idx_train[0], idx_train[1]] = 0
    idx_test = np.array(np.where(y_test == -1))
    y_test[idx_test[0], idx_test[1]] = 0

    return y_train, y_test


def extract_40attr_vectors(dataset, input_dim, modelname, model):
    print('-------------------extracting 40-dimensional attribute vectors from the model')
    if dataset == 'lfw':
        from load_data import load_lfw_aligned
        feat_save_dir = "/data/jihyec/matconvnet-fresh-compile/examples/robust_faceRecog_on_Gogs/" \
                        "step3_extract_attr_on_vgg_easy_set/face_verification/lfw_classified_attr/%s" % modelname

        import os
        if not os.path.exists(feat_save_dir):
            os.makedirs(feat_save_dir)

        # load data
        view = 1
        X_train1, X_train2, X_test1, X_test2 = load_lfw_aligned(image_dim=input_dim, ith_fold=0, view=view)

        # extract 40attr_vectors
        feat_train1 = model.predict(X_train1)
        feat_train2 = model.predict(X_train2)
        feat_test1 = model.predict(X_test1)
        feat_test2 = model.predict(X_test2)

        feat_save_file = "%s/lfw_view1_dim%d_40attr_classified.mat" % (feat_save_dir, input_dim)
        scipy.io.savemat(feat_save_file, mdict={'feat_train1': feat_train1, 'feat_train2': feat_train2,
                                                'feat_test1': feat_test1, 'feat_test2': feat_test2})

        # load data
        view = 2
        for fold in range(1, 11):  # 10 fold cross-validation
            X_eval1, X_eval2 = load_lfw_aligned(image_dim=input_dim, ith_fold=fold, view=view)

            # extract 40attr_vectors
            feat_eval1 = model.predict(X_eval1)
            feat_eval2 = model.predict(X_eval2)

            feat_save_file = "%s/lfw_view2_dim%d_fold%d_40attr_classified.mat" % (feat_save_dir, input_dim, fold)
            scipy.io.savemat(feat_save_file, mdict={'feat_eval1': feat_eval1, 'feat_eval2': feat_eval2})


def add_feat_eng_and_fcl(model):
    from keras.models import Model
    from keras.layers import Dense, Activation
    from keras.layers.normalization import BatchNormalization
    from keras.initializers import Constant

    feat_eng_weightpath = '/data/jihyec/matconvnet-fresh-compile/examples/robust_faceRecog_on_Gogs/' \
                          'step3_extract_attr_on_vgg_easy_set/data/easy_set2_0.8_feat_engineering_weights.mat'
    loaded_feat = scipy.io.loadmat(feat_eng_weightpath)
    weights = loaded_feat['weights'][0]
    feat = Dense(units=40*20, activation=None, kernel_initializer=Constant(np.zeros((40, 40*20))), use_bias=False)(model.layers[-1].output)
    fcl1 = Activation('relu')(BatchNormalization()(Dense(units=40*20, activation=None)(feat)))
    fcl2 = Activation('relu')(BatchNormalization()(Dense(units=40*20, activation=None)(fcl1)))
    fcl3 = Activation('softmax')(BatchNormalization()(Dense(units=20, activation=None)(fcl2)))

    model = Model(model.layers[0].input, fcl3)

    w = model.layers[-10].get_weights()
    weights = weights[0][0][0]
    for ith_sub in range(0, 20):  # 0 ~ 19
        start = ith_sub * 40  # 0, 40, 80, ...
        for ith_w in range(0, 40):  # 0 ~ 39
            w[0][ith_w, start + ith_w] = weights[ith_w, ith_sub]
    model.layers[-10].set_weights(weights=w)

    return model

