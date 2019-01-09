from __future__ import division

import argparse
import sys
import os
import scipy
import numpy as np
from keras.models import load_model, Model
from keras.layers import Input, Dense, BatchNormalization, Activation
from keras.optimizers import SGD
from keras.utils import to_categorical
from eval_utils import compute_attr_embedding, l2_normalize, compute_embedding, evaluate_verification, plot_roc_general
import lfw


def prepare_attr_classifier(args):
    # load attribute classifier
    if args.custom_loss:
        from train_utils import binary_crossentropy_custom
        import keras.activations
        keras.activations.custom_activation = binary_crossentropy_custom
        model = load_model(args.model_path,
                           custom_objects={'binary_crossentropy_custom': binary_crossentropy_custom})
    else:
        model = load_model(args.model_path)

    return model


def pairs_to_feat(path_to_pairs, data_dir, file_ext, model):
    # path to LFW images to test verification performance
    pairs = lfw.read_pairs(os.path.expanduser(path_to_pairs))
    paths, true_issame = lfw.get_paths(os.path.expanduser(data_dir), pairs, file_ext)  # paths: (1,12000) array, true_issame: (1,6000)

    # compute embeddings
    # emb_all = compute_embedding(paths, model, args.lfw_batch_size) # (1, Nx2) array
    emb_all = compute_attr_embedding(model, paths)
    emb1 = emb_all[0::2]
    emb2 = emb_all[1::2]
    # perform feature engineering
    # refer to: Kumar, Neeraj, et al. "Attribute and simile classifiers for face verification." ICCV 09.
    g = scipy.stats.norm(0, 1).pdf((emb1 + emb2)/2)
    feat1 = np.multiply(np.absolute(emb1 - emb2), g)
    feat2 = np.multiply(np.multiply(emb1, emb2), g)
#    feat = np.concatenate((feat1, feat2), axis=1)
    feat = feat1

    # test
    # scipy.io.savemat('./test.mat', mdict={'emb1': emb1, 'emb2':emb2, 'g': g, 'feat': feat}) 

    return feat, to_categorical(true_issame*1)




def prepare_verifier(attr_dim, num_fcl):
    # create model for  verification
    input = Input(shape=(attr_dim,))
    fcl = Activation('relu', name='verification_relu1') \
          (BatchNormalization(name='verification_batchnorm1') \
          (Dense(attr_dim, activation=None, name='verification_fcl1')(input)))

    for l in range(2, num_fcl):
        fcl = Activation('relu', name="verification_relu%d" % l) \
             (BatchNormalization(name="verification_batchnorm%d" % l) \
             (Dense(attr_dim, activation=None, name="verification_fcl%d" % l)(fcl)))
    fcl = Activation('softmax', name='verification') \
          (BatchNormalization(name="verification_batchnorm%d" % num_fcl) \
          (Dense(2, activation=None, name="verification_fcl%d" % num_fcl)(fcl)))

    verifier = Model(inputs=input, outputs=fcl)
    # verifier.summary()

    # train verification layers
    sgd = SGD(lr=0.0001, momentum=0.9)
    verifier.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])
    return verifier


def evaluate_verification_attr(X, y, num_layer, num_folds=10, num_epoch=100):
    num_pairs = min(len(y), X.shape[0])
    tprs = np.zeros((num_folds))
    fprs = np.zeros((num_folds))
    accurs = np.zeros((num_folds))
    thresh = np.zeros((num_folds))
    
    indices = np.arange(num_pairs)    

    from sklearn.model_selection import KFold
    k_fold = KFold(n_splits=num_folds, shuffle=False)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Train the verifier with train set
        verifier = prepare_verifier(attr_dim=40, num_fcl=num_layer)
        verifier.fit(X[train_set], y[train_set], batch_size=50, epochs=num_epoch, shuffle=True, validation_split=0.2)

        # Evaluate with test set
        y_pred = verifier.predict(X[test_set])
        # y_pred = np.array(np.argmax(y_pred,axis=1))
        # y_test = np.array(np.argmax(y[test_set],axis=1))
        y_test = y[test_set]

        accurs[fold_idx] = np.sum(np.equal(y_test[:,1], np.array(np.argmax(y_pred,axis=1)))) / len(y_pred)
        thresh[fold_idx], fprs[fold_idx], tprs[fold_idx] = plot_roc_general(y_test[:,1], y_pred[:,1], savefig=None)

    return thresh, accurs, fprs, tprs


def main(args):

    model = prepare_attr_classifier(args)

    """
    ############### evaluate on LFW view 1
    # prepare feature data for training verifier
    X_train, y_train = pairs_to_feat(path_to_pairs=os.path.join(args.lfw_pairs_dir,'lfw-view1_pairsDevTrain.txt'), data_dir=args.lfw_dir, file_ext=args.lfw_file_ext, model=model)

    # train verification layers
    verifier = prepare_verifier(attr_dim=40, num_fcl=args.num_layer)
    verifier.fit(X_train, y_train, batch_size=50, epochs=10, shuffle=True, validation_split=0.2)


    # prepare feature data for testing verifier
    X_test, y_test = pairs_to_feat(path_to_pairs=os.path.join(args.lfw_pairs_dir,'lfw-view1_pairsDevTest.txt'), data_dir=args.lfw_dir, file_ext=args.lfw_file_ext, model=model) # y_test: (1000, 2)


    # evalute the performance on test set
    y_pred = verifier.predict(X_test)
    y_pred = np.array(np.argmax(y_pred,axis=1))
    y_test = np.array(np.argmax(y_test,axis=1))
   
    accuracy = np.sum(np.equal(y_test, y_pred)) / len(y_pred)    
    print('Accuracy: %1.3f' % accuracy)
    thresh, fpr, tpr = plot_roc_general(y_test, y_pred, savefig="../model/result/verification_attr_fcl%d.png" %args.num_layer)
    print('Threshold at EER point: %1.3f' % thresh)
    print('FPR at EER point: %1.3f' % fpr)
    print('TPR at EER point: %1.3f' % tpr)
    """

    ############### evaluate on LFW view 2
    # prepare feature data for evaluation verifier
    X, y = pairs_to_feat(path_to_pairs=os.path.join(args.lfw_pairs_dir,'lfw-view2_pairs.txt'), data_dir=args.lfw_dir, file_ext=args.lfw_file_ext, model=model)
    thresh, accurs, fprs, tprs = evaluate_verification_attr(X, y, args.num_layer, num_folds=10, num_epoch=100)
    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accurs), np.std(accurs)))
    print('Threshold: %1.3f+-%1.3f' % (np.mean(thresh), np.std(thresh)))
    print('FPR: %1.3f+-%1.3f' % (np.mean(fprs), np.std(fprs)))
    print('TPR: %1.3f+-%1.3f' % (np.mean(tprs), np.std(tprs)))



def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--lfw_dir', type=str, help='Path to the data directory containing aligned LFW data.',
                        default='/data/jihyec/data/lfw/lfw-deepfunneled-dim160')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='jpg', choices=['jpg', 'png'])
    parser.add_argument('--lfw_pairs_dir', type=str,
        help='Path to directory where lfw-view1_pairsDevTrain.txt, lfw-view1_pairsDevTest.txt, lfw-view2_pairs.txt are located.', default='../data')
    parser.add_argument('--lfw_batch_size', type=int, help='Number of images to process in a batch in the LFW test set.', default=600)
    parser.add_argument('--model_path', type=str, help='Path to the trained attribute classifier.',
                        default='/data/jihyec/adversarial-attack/model/VGG16-dim160-attrClassifier.h5')
    parser.add_argument('--save_verifier', type=str, help='Path to save the trained face verifier.', default='/data/jihyec/adversarial-attack/model/VGG16-dim160-attr-verifier.h5')
    parser.add_argument('--feat_eng', type=str2bool, help='Whether to perform feature engineering with attribute vectors', default=True)
    parser.add_argument('--num_layer', type=int, help='Number of fully connected layers consisting of face verifier', default=2)
    parser.add_argument('--custom_loss', type=str2bool, help='Whether the attrClassifier trained with custom loss', default=True)

    return parser.parse_args(argv)


def str2bool(value):
    return value.lower == 'true'


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
