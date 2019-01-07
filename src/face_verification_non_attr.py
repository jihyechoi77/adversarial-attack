# reference: https://github.com/davidsandberg/facenet/blob/master/src/validate_on_lfw.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import numpy as np
import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

from keras.models import load_model
import lfw
from eval_utils import compute_attr_embedding
import facenet

def prepare_model(args):

    if args.model_type == 'facenet':
        # load facenet model trained with MS-CelebA-1M dataset
        # keras model converted from tf model in https://github.com/davidsandberg/facenet
        model = load_model('../model/model-20180402-114759.h5') 

    # elif args.model_type == 'vgg16':

    # model.summary()
    return model

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def compute_embedding(paths, model, batch_size):
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')
    num_images = len(paths)
    num_batches = int(math.ceil(1.0*num_images / batch_size))
    emb_array = np.zeros((num_images, model.layers[-1].output_shape[-1]))

    # load images and compute embeddings
    for i in range(num_batches):
        start_idx = i*batch_size
        end_idx = min((i+1)*batch_size, num_images)
        paths_batch = paths[start_idx:end_idx]
        images = facenet.load_data(paths_batch, False, False, image_size=160)
         # test
#        import scipy.misc
#        scipy.misc.imsave('./test.jpg',images[1])

        emb_array[start_idx:end_idx,:] = l2_normalize(model.predict(images))
        
        # test
#       from scipy.spatial import distance
#       print(distance.euclidean(emb_array[start_idx], emb_array[start_idx+1]))

    return emb_array


def main(args):
    
    # load model to extract non-attribute based embeddings of face images
    model = prepare_model(args)

    # path to LFW images to test verification performance
    pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
    paths, true_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)  # paths: (1,12000) array, true_issame: (1,6000)
#    print(np.shape(paths))
#    print(np.shape(true_issame))
#    print(true_issame[298:302])   # T, T, F, F
    # compute embeddings
    emb_all = compute_embedding(paths, model, args.lfw_batch_size) # (1, Nx2) array

    # evalute verification performance
    tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(emb_all, true_issame, nrof_folds=args.lfw_nrof_folds)
    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.3f' % eer)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--lfw_dir', type=str, help='Path to the data directory containing aligned LFW data.',default='/data/jihyec/data/lfw/lfw-deepfunneled-mtcnnpy_dim160')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='jpg', choices=['jpg', 'png'])
    parser.add_argument('--lfw_batch_size', type=int, help='Number of images to process in a batch in the LFW test set.', default=600)
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='../data/lfw-view2_pairs.txt')
    parser.add_argument('--lfw_nrof_folds', type=int, help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--model_type', type=str, help='Type of non-attribute based model.', default='facenet')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
