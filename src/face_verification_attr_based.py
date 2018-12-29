from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from keras.layers import Input, Dense, BatchNormalization, Activation
from keras.models import Model

import numpy as np
import scipy.io
import scipy.misc
from scipy.io import loadmat, savemat
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import keras
import keras.backend as K

from eval_utils import compute_accuracy, plot_roc_general
from evaluate_multilab_classifiers import load_vgg_model, load_inception_model
from train_utils import extract_40attr_vectors
from attack_utils import apply_fgsm_np, apply_CarliniWagnerL2_np



def load_data(datafile, dataset):
    data = loadmat(datafile)
    Y_train = np_utils.to_categorical(data['Y_train'])
    y_test = data['Y_test']
    X_train = data['X_train']
    X_test = data['X_test']

    # X_train = X_train[:, 40:]
    # X_test = X_test[:, 40:]

    if dataset is 'vgg':
        Y_subj_test = data['Y_subj_test']
        return (X_train, Y_train), (X_test, y_test), Y_subj_test
    else:  # for lfw and pubfig dataset
        Y_test = np_utils.to_categorical(y_test)
        return (X_train, Y_train), (X_test, Y_test)


def create_model(input_dim, layer_depth, input):
    # layer_depth: number of fully-connected layers
    # input: input tensor to the fcl layers
    # input = Input(shape=(input_dim, ))
    fcl = Activation('relu', name='verification_activation1')(BatchNormalization(name='verification_batchnorm1')
                                                              (Dense(input_dim, activation=None, name='verification_fcl1')(input)))
    for l in range(2, layer_depth):
        fcl = Activation('relu', name="verification_activation%d" % l)(BatchNormalization(name="verification_batchnorm%d" % l)
                                                                       (Dense(input_dim, activation=None, name="verification_fcl%d" % l)(fcl)))
    fcl = Activation('softmax', name='verification')(BatchNormalization(name="verification_batchnorm%d" % layer_depth)
                                                     (Dense(2, activation=None, name="verification_fcl%d" % layer_depth)(fcl)))

    return fcl
    # model = Model(inputs=input, outputs=fcl)
    # return model


def verify_on_vgg(feat_eng):
    if feat_eng:
        datafile = '/data/jihyec/matconvnet-fresh-compile/examples/robust_faceRecog_on_Gogs/' \
                   'step3_extract_attr_on_vgg_easy_set/face_verification/data/easy_set_data_feat_engineer_for_verification.mat'
        resultfile = '/data/jihyec/matconvnet-fresh-compile/examples/robust_faceRecog_on_Gogs/' \
                     'step3_extract_attr_on_vgg_easy_set/face_verification/easy_set_data_feat_engineer_verif_results.mat'
        input_dim = 40*20
    else:
        datafile = '/data/jihyec/matconvnet-fresh-compile/examples/robust_faceRecog_on_Gogs/' \
                   'step3_extract_attr_on_vgg_easy_set/face_verification/data/easy_set_data_for_verification.mat'
        resultfile = '/data/jihyec/matconvnet-fresh-compile/examples/robust_faceRecog_on_Gogs/' \
                     'step3_extract_attr_on_vgg_easy_set/face_verification/easy_set_data_verif_results.mat'
        input_dim = 40

    (X_train, Y_train), (X_test, y_test), Y_subj_test = load_data(datafile=datafile, dataset='vgg')
    model = create_model(input_dim=input_dim, layer_depth=3)

    model.fit(x=X_train, y=Y_train, batch_size=128, epochs=20)

    pred = model.predict(x=X_test)
    pred = np.transpose([np.array(np.argmax(pred,axis=1))])
    accur = compute_accuracy(truth=y_test, pred=pred)
    print("verification accuracy on test data: %0.3f" % accur)

    savefig = '/data/jihyec/matconvnet-fresh-compile/examples/robust_faceRecog_on_Gogs/' \
                   'step3_extract_attr_on_vgg_easy_set/face_verification/ROC_curve_verifier.png'
    plot_roc_general(truth=y_test, pred=pred, savefig=savefig)

    confmat = confusion_matrix(y_test, pred)
    scipy.io.savemat(resultfile, mdict={'truth': y_test, 'pred':pred, 'confmat': confmat, 'subj_label': Y_subj_test})


def verify_on_lfw(modelname, feat_eng, load_verifier):
    if feat_eng:
        datafile = '/data/jihyec/matconvnet-fresh-compile/examples/robust_faceRecog_on_Gogs/' \
                   'step3_extract_attr_on_vgg_easy_set/face_verification/data/lfw_view1_attrdata_feat_engineer_for_verification.mat'
        (X_train, Y_train), (X_val, Y_val) = load_data(datafile=datafile, dataset='lfw')
        input_dim = np.shape(X_train)[1] * 20

        resultFile = '/data/jihyec/matconvnet-fresh-compile/examples/robust_faceRecog_on_Gogs/' \
                     'step3_extract_attr_on_vgg_easy_set/face_verification/lfw_view1_attrdata_feat_engineer_verif_results.mat'

    else:
        datafile = "/data/jihyec/matconvnet-fresh-compile/examples/robust_faceRecog_on_Gogs/" \
                   "step3_extract_attr_on_vgg_easy_set/face_verification/data/%s/" \
                   "lfw_view1_attrdata_Kumar_ver_without_g_for_verification.mat" % modelname
        (X_train, Y_train), (X_val, Y_val) = load_data(datafile=datafile, dataset='lfw')
        input_dim = int(np.shape(X_train)[1])

        resultFile = "/data/jihyec/matconvnet-fresh-compile/examples/robust_faceRecog_on_Gogs/" \
                     "step3_extract_attr_on_vgg_easy_set/face_verification/lfw_view2_without_g_verif_results_%s.mat" % modelname

    input = Input(shape=(input_dim,))
    # model = create_model(input_dim=input_dim, layer_depth=2, input=input)
    fcl = create_model(input_dim=input_dim, layer_depth=2, input=input)
    model = Model(inputs=input, outputs=fcl)
    model.summary()
    model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])

    ################## train the network
    # verifierPath = "/data/jihyec/vgg_keras_tf/trained_models/%s_verifier.h5" % modelname
    verifierPath = "/home/jihyec/adversarial-ml/trained_models/%s_verifier.h5" % modelname
    if load_verifier:
        model.load_weights(verifierPath)
    else:
        from keras import callbacks
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=2, mode='auto')
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=2, mode='auto',
                                                epsilon=0.001, cooldown=0, min_lr=0)
        model.fit(x=X_train, y=Y_train, batch_size=100, epochs=100, validation_data=(X_val, Y_val), callbacks=[early_stopping, reduce_lr], shuffle=True)
        model.save(verifierPath)
    accuracy = []
    for ith_fold in range(1, 11):  # 10 fold cross-validation

        testfile = "/data/jihyec/matconvnet-fresh-compile/examples/robust_faceRecog_on_Gogs/" \
                   "step3_extract_attr_on_vgg_easy_set/face_verification/data/%s/" \
                   "lfw_view2_fold%d_attrdata_Kumar_ver_without_g_for_verification" % (modelname, ith_fold)
        data = loadmat(testfile)
        X_test = data['X_test']
        # X_test = X_test[:, 40:]
        y_test = data['Y_test']

        pred = model.predict(x=X_test)
        pred = np.transpose([np.array(np.argmax(pred, axis=1))])
        accur = compute_accuracy(truth=y_test, pred=pred)
        print("verification accuracy on %d th fold test data: %0.3f" % (ith_fold, accur))
        accuracy.append(accur)
        confmat = confusion_matrix(y_test, pred)
        FPR = confmat[0,1] / (confmat[0,0] + confmat[0,1])
        TPR = confmat[1,1] / (confmat[1,0] + confmat[1,1])
        print("TPR = %0.3f, FPR = %0.3f \n" % (TPR, FPR))

    print("mean of verification accuracy on LFW view2 dataset: %0.6f" % np.mean(accuracy))
    print("variance of verification accuracy on LFW view2 dataset: %0.6f" % np.var(accuracy))
    #savefig = "/data/jihyec/matconvnet-fresh-compile/examples/robust_faceRecog_on_Gogs/" \
    #               "step3_extract_attr_on_vgg_easy_set/face_verification/%s_verif_on_lfw_view2_ROC_curve.png" % modelname
    #plot_roc_general(truth=y_test, pred=pred, savefig=savefig)
    scipy.io.savemat(resultFile, mdict={'truth': y_test, 'pred': pred, 'confmat': confmat})

    return model


def load_lfw_ver2_subdata(modelname, input_dim, ith_fold, identical, model):
    from load_data import load_lfw_aligned
    X_eval1, X_eval2 = load_lfw_aligned(image_dim=input_dim, ith_fold=ith_fold, view=2)
    N = len(X_eval1)  # 0~N/2-1 : identical, N/2~(N-1) : different
    # import scipy.misc
    # scipy.misc.imsave('/data/jihyec/vgg_keras_tf/identical1.jpg', X_eval1[N/2-1])
    # scipy.misc.imsave('/data/jihyec/vgg_keras_tf/identical2.jpg', X_eval2[N / 2 - 1])
    # scipy.misc.imsave('/data/jihyec/vgg_keras_tf/different1.jpg', X_eval1[N / 2])
    # scipy.misc.imsave('/data/jihyec/vgg_keras_tf/different2.jpg', X_eval2[N / 2])

    testfile = "/data/jihyec/matconvnet-fresh-compile/examples/robust_faceRecog_on_Gogs/" \
               "step3_extract_attr_on_vgg_easy_set/face_verification/data/%s/" \
               "lfw_view2_fold%d_attrdata_Kumar_ver_without_g_for_verification" % (modelname, ith_fold)
    data = loadmat(testfile)
    # feat1_all = data['feat_eval1']
    # feat2_all = data['feat_eval2']

    # extract 40attr_vectors
    feat1_all = model.predict(X_eval1)
    feat2_all = model.predict(X_eval2)

    if identical:
        subject_X = X_eval1[:int(N/2)]
        # subject_attr = feat1_all[:N/2]
        target_X = X_eval2[:int(N/2)]
        target_attr = feat2_all[:int(N/2)]
    else:
        subject_X = X_eval1[int(N/2):]
        subject_X = X_eval1[int(N/2):]
        # subject_attr = feat1_all[N/2:]
        target_X = X_eval2[int(N/2):]
        target_attr = feat2_all[int(N/2):]
    # return subject_X, subject_attr, target_attr
    return subject_X, target_X, target_attr


def verify_on_pubfig(feat_eng):
    if feat_eng:
        datafile = '/data/jihyec/matconvnet-fresh-compile/examples/robust_faceRecog_on_Gogs/' \
                   'step3_extract_attr_on_vgg_easy_set/face_verification/data/pubfig_data_feat_engineer_for_verification.mat'
        resultfile = '/data/jihyec/matconvnet-fresh-compile/examples/robust_faceRecog_on_Gogs/' \
                     'step3_extract_attr_on_vgg_easy_set/face_verification/pubfigeval_feat_engineer_verif_results.mat'
        input_dim = 40*20
    else:
        datafile = '/data/jihyec/matconvnet-fresh-compile/examples/robust_faceRecog_on_Gogs/' \
                   'step3_extract_attr_on_vgg_easy_set/face_verification/data/pubfig_data_for_verification.mat'
        resultfile = '/data/jihyec/matconvnet-fresh-compile/examples/robust_faceRecog_on_Gogs/' \
                     'step3_extract_attr_on_vgg_easy_set/face_verification/pubfigeval_verif_results.mat'
        input_dim = 40

    (X_train, Y_train), (X_test, y_test) = load_data(datafile=datafile, dataset='pubfig')
    model = create_model(dim=input_dim)

    model.fit(x=X_train, y=Y_train, batch_size=128, epochs=50)

    pred = model.predict(x=X_test)
    pred = np.transpose([np.array(np.argmax(pred, axis=1))])
    accur = compute_accuracy(truth=y_test, pred=pred)
    print("verification accuracy on test data: %0.3f" % accur)

    savefig = '/data/jihyec/matconvnet-fresh-compile/examples/robust_faceRecog_on_Gogs/' \
                   'step3_extract_attr_on_vgg_easy_set/face_verification/ROC_curve_verifier_on_pubfigeval.png'
    plot_roc_general(truth=y_test, pred=pred, savefig=savefig)

    confmat = confusion_matrix(y_test, pred)


def build_end_to_end_verifier(attrClassifier, target_attr, modelname):
    from custom_layer import FeatEngineerKumarVer
    attr = FeatEngineerKumarVer(input_shape=np.shape(target_attr), target_attr=target_attr)(attrClassifier.layers[-1].output)
    out = create_model(input_dim=80, layer_depth=2, input=attr)
    model = Model(inputs=attrClassifier.layers[0].input, outputs=out)

    # model.summary()

    # verifierPath = "/data/jihyec/vgg_keras_tf/trained_models/%s_verifier.h5" % modelname
    verifierPath = "/home/jihyec/adversarial-ml/vgg_keras_tf/trained_models/%s_verifier.h5" % modelname
    model.load_weights(verifierPath, by_name=True)

    return model


def apply_verif_attack(sess, attrClassifier, modelname, subject_X, target_X, target_attr, truth):
    # pred = np.empty((0, 11), float)
    accur = np.zeros((1, 11), float)
    for idx in range(0, np.shape(subject_X)[0]):
        X_val = np.expand_dims(subject_X[idx], axis=0)
        X_tar = target_X[idx]

        model = build_end_to_end_verifier(attrClassifier, target_attr=target_attr[idx], modelname=modelname)
        # X_adv = apply_fgsm_np(sess, model, X_val, Y_test=truth, targeted=False, visualize=False, nb_classes=2)
        X_adv = apply_CarliniWagnerL2_np(sess, model, X_val, Y_test=truth, targeted=False, visualize=True, nb_classes=2)

        # orig_lab = model.predict(X_val)
        pred = model.predict(X_adv)
        new_lab = np.transpose(np.argmax(pred, axis=1))
        # pred = np.vstack((pred, new_lab))
        accur += (new_lab == np.argmax(truth))

        """
        # save the images
        scipy.misc.imsave("/data/jihyec/vgg_keras_tf/attack/adv_images/target%d.png" % idx, X_tar)
        eps = [0, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
        for i in range(0, len(eps)):
            scipy.misc.imsave("/data/jihyec/vgg_keras_tf/attack/adv_images/adv%d_eps_%0.3f_lab_%d_pred_%0.3f.png"
                              % (idx, eps[i], new_lab[i], pred[i, new_lab[i]]), X_adv[i])
        """
        del model

    accur /= np.shape(subject_X)[0]
    return accur


def main(argv=None):
    sess = tf.Session()
    keras.backend.set_session(sess)
    K.set_learning_phase(0)  # set learning phase

    feat_eng = False
    modelname = 'VGG16_multilab_using_customloss'  # modeltype: VGG16s, VGG16_multilab, InceptionResnetV2_multilab,
    if 'VGG16' in modelname:
        attrClassifier = load_vgg_model()
    else:
        attrClassifier = load_inception_model()
    # attrClassifier.load_weights("/data/jihyec/vgg_keras_tf/trained_models/%s_notop_on_celeba.h5" % modelname) # for grey4
    attrClassifier.load_weights("/home/jihyec/adversarial-ml/trained_models/%s_notop_on_celeba.h5" % modelname) # for devbox

    # extract_40attr_vectors(dataset='lfw', input_dim=140, modelname=modelname, model=attrClassifier)
    # verifier = verify_on_lfw(modelname=modelname, feat_eng=feat_eng, load_verifier=True)


    ####################################################################################### apply attacks
    num_folds = 10
    accur_pos = np.empty((0, num_folds+1), float)
    accur_neg = np.empty((0, num_folds+1), float)
    accur = np.empty((0, num_folds+1), float)
    for ith_fold in range(1, num_folds+1):  # 1, 11
        print("------%d th fold" % ith_fold)
        print("positive pairs")
        subject_X, target_X, target_attr = load_lfw_ver2_subdata(modelname, 140, ith_fold, identical=True, model=attrClassifier)
        # accur_pos = np.vstack((accur_pos, apply_verif_attack(sess, attrClassifier, modelname, subject_X, target_X, target_attr, truth=[[0, 1]])))
        accur_pos = apply_verif_attack(sess, attrClassifier, modelname, subject_X, target_X, target_attr, truth=[[0, 1]])

        print("negative pairs")
        subject_X, target_X, target_attr = load_lfw_ver2_subdata(modelname, 140, ith_fold, identical=False, model=attrClassifier)
        # accur_neg = np.vstack((accur_neg, apply_verif_attack(sess, attrClassifier, modelname, subject_X, target_X, target_attr, truth=[[1, 0]])))
        accur_neg = apply_verif_attack(sess, attrClassifier, modelname, subject_X, target_X, target_attr, truth=[[1, 0]])

        # accur = np.vstack((accur, (accur_pos + accur_neg)/2))
        accur = (accur_pos + accur_neg) / 2

        #result_save_file = "/data/jihyec/matconvnet-fresh-compile/examples/robust_faceRecog_on_Gogs/" \
        #                   "step3_extract_attr_on_vgg_easy_set/face_verification/lfw_view2_fold%d_without_g_attack_results_%s.mat" % (ith_fold, modelname)
        result_save_file = "/home/jihyec/adversarial-ml/results/lfw_view2_fold%d_without_g_attack_results_%s.mat" % (ith_fold, modelname)
        savemat(result_save_file, mdict={'accur': accur, 'accur_pos': accur_pos, 'accur_neg': accur_neg})


    """

    # for debugging
    pred = []
    for idx in range(0, np.shape(subject_X)[0]):
        model = build_end_to_end_verifier(attrClassifier, target_attr=target_attr[idx], modelname=modelname)
        # intermediate_layer_model0 = Model(inputs=model.layers[0].input,
        #                                  outputs=model.layers[-8].output)
        intermediate_layer_model = Model(inputs=model.layers[0].input,
                                         outputs=model.layers[-7].output)
        # attr_40dim = intermediate_layer_model0.predict(np.expand_dims(subject_X[idx], axis=0))
        attr_80dim = intermediate_layer_model.predict(np.expand_dims(subject_X[idx], axis=0))
        ith_pred = verifier.predict(attr_80dim)
        pred = np.append(pred, np.argmax(ith_pred, axis=1))  # 0: different, 1: identical
        del model

    accur = np.sum(pred == 1) / len(pred)  # verification accuracy
    print(accur)
    """

if __name__ == '__main__':
    main()

