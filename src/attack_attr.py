# Original file: https://github.com/tensorflow/cleverhans/blob/master/examples/facenet_adversarial_faces/facenet_fgsm.py

from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
import random
import argparse
import math
from sklearn.model_selection import KFold
from keras import backend as K
from cleverhans.model import Model
from cleverhans.attacks import FastGradientMethod, CarliniWagnerL2, Noise
from cleverhans.utils_tf import model_eval
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Model as KModel
import facenet
import lfw
from attack_utils import run_attack, evaluate


def load_images(path1, path2):

    """
    # Load images
    faces1 = facenet.load_data(path1, False, False, image_size=160)
    faces2 = facenet.load_data(path2, False, False, image_size=160)

    # Change pixel values to 0 to 1 values
    min_pixel = min(np.min(faces1), np.min(faces2))
    max_pixel = max(np.max(faces1), np.max(faces2))
    faces1 = (faces1 - min_pixel) / (max_pixel - min_pixel)
    faces2 = (faces2 - min_pixel) / (max_pixel - min_pixel)
    """

    from keras.preprocessing.image import load_img, img_to_array
    faces1 = np.empty((len(path1), 160, 160, 3))
    faces2 = np.empty((len(path2), 160, 160, 3))
    for i, ID in enumerate(path1):
        # Store sample
        faces1[i,] = img_to_array(load_img(ID)) / 255
        faces2[i,] = img_to_array(load_img(path2[i])) / 255

    return faces1, faces2


class AttributeModel(Model):
    model_path = "/data/jihyec/adversarial-attack/model/VGG16-dim160-attrClassifier.h5"
    verifier_path = "/data/jihyec/adversarial-attack/model/VGG16-dim160-attr-verifier-fcl2.h5"
    custom_loss = True

    def __init__(self):
        super(AttributeModel, self).__init__(scope='model')

        # load attribute classifier
        if self.custom_loss:
            from train_utils import binary_crossentropy_custom
            import keras.activations
            keras.activations.custom_activation = binary_crossentropy_custom
            model = load_model(self.model_path,
                               custom_objects={'binary_crossentropy_custom': binary_crossentropy_custom})
        else:
            model = load_model(self.model_path)

        self.face_input = model.input
        self.embedding_output = model.output

    def convert_to_classifier(self):
        # Create victim_embedding placeholder
        self.victim_embedding_input = tf.placeholder(tf.float32, shape=(None, 40),
                                                     name="victim_embedding_input")

        # Feature engineering given input(adv) and victim embeddings
        feat = tf.abs(self.embedding_output - self.victim_embedding_input)

        # Load verifier trained on LFW View1
        K.set_learning_phase(False)  # inference not training
        verifier = load_model(self.verifier_path)
        logit_layer_model = KModel(inputs=verifier.input, outputs=verifier.layers[-1].input)
        self.logits = logit_layer_model(feat)
        self.softmax_output = verifier(feat)

        # Save softmax layer
        self.layer_names = []
        self.layers = []
        self.layers.append(self.logits)
        self.layer_names.append('logits')
        self.layers.append(self.softmax_output)
        self.layer_names.append('probs')  # 'probs' not 'logits'

    def fprop(self, x, set_ref=False):
        return dict(zip(self.layer_names, self.layers))


def prepare_attack(sess, args, model, adv_input, target_embeddings):
    if args.attack_type == 'FGSM':
        # Define FGSM for the model
        steps = 1
        alpha = args.eps / steps
        fgsm = FastGradientMethod(model)
        fgsm_params = {'eps': alpha,
                       'clip_min': 0.,
                       'clip_max': 1.}
        adv_x = fgsm.generate(model.face_input, **fgsm_params)
    elif args.attack_type == 'CW':
        model.face_input.set_shape(np.shape(adv_input))
        # Instantiate a CW attack object
        cw = CarliniWagnerL2(model, sess)
        cw_params = {'binary_search_steps': 1,  # 1
                     'max_iterations': 100,  # 100
                     'learning_rate': .2,  # .2
                     'batch_size': args.lfw_batch_size,
                     'initial_const': args.init_c, # 10
                     'confidence': 10}
        #              # model.batch_size: 10, model.phase_train: False}
        feed_dict = {model.face_input: adv_input, model.victim_embedding_input: target_embeddings}
        #              # model.batch_size: 10, model.phase_train: False}
        # adv_x = cw.generate(model.face_input, feed_dict, **cw_params)
        adv_x = cw.generate(model.face_input, **cw_params)
        # adv_x = cw.generate_np(adv_input, **cw_params)
        print('hello')
    elif args.attack_type == 'random':
        random_attack = Noise(model, sess)
        noise_params = {'eps': args.eps,
                        'ord': np.inf,
                        'clip_min': 0, 'clip_max': 1}
        adv_x = random_attack.generate(model.face_input, **noise_params)

    return adv_x


def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:

            # Load model
            model = AttributeModel()
            # Convert to classifier
            model.convert_to_classifier()

            # Load images paths and labels
            pairs = lfw.read_pairs(args.lfw_pairs)
            paths, true_issame = lfw.get_paths(args.lfw_dir, pairs)
            path1_all = paths[0::2]
            path2_all = paths[1::2]
            labels_all = 1 * np.array(true_issame)  # (N, ) array of 0 of 1, N = 6000 for LFW view2 dataset
            # NOTE! for this example, 1: identical, 0: different

            k_fold = KFold(n_splits=10, shuffle=False)  # 10-fold cross validation
            indices = np.arange(np.shape(path1_all)[0])
            real_labels = []
            adversarial_labels = []
            labels_evaluated = []

            for ith_fold, (_, fold_idx) in enumerate(k_fold.split(indices)):
                if ith_fold == args.lfw_nrof_folds:
                    break

                fold_idx = np.append(fold_idx[:50], fold_idx[-50:])

                path1 = [path1_all[i] for i in fold_idx]
                path2 = [path2_all[i] for i in fold_idx]
                labels_evaluated = np.append(labels_evaluated, [labels_all[i] for i in fold_idx], axis=-1)
                # labels = labels_all[fold_idx]
                num_images = len(path1)
                num_batches = int(math.ceil(1.0 * num_images / args.lfw_batch_size))

                for i in range(num_batches):
                    start_idx = i * args.lfw_batch_size
                    end_idx = min((i + 1) * args.lfw_batch_size, num_images)

                    path1_batch = path1[start_idx:end_idx]
                    path2_batch = path2[start_idx:end_idx]
                    # labels_batch = to_categorical(labels[start_idx:end_idx], num_classes=2)

                    # Load pairs of faces and their labels in one-hot encoding
                    adv_faces, target_faces = load_images(path1_batch, path2_batch)

                    # Create target embeddings using Facenet itself
                    feed_dict = {model.face_input: target_faces} #, model.phase_train: False}
                    target_embeddings = sess.run(model.embedding_output, feed_dict=feed_dict)

                    # Run attack
                    feed_dict = {model.face_input: adv_faces, model.victim_embedding_input: target_embeddings} #,
                                 #model.batch_size: 10, model.phase_train: False}
                    adv_x = prepare_attack(sess, args, model, adv_faces, target_embeddings)
                    real_labels_batch, adversarial_labels_batch = run_attack(sess, model, adv_x, feed_dict)

                    """
                # Define input TF placeholder
                x = tf.placeholder(tf.float32, shape=(None, 160, 160, 3), name='x')
                y = tf.placeholder(tf.float32, shape=(None, 2), name='y')
                preds = model.softmax_output
  
                # Evaluate the accuracy on legitimate test examples
                feed_dict = {model.face_input: adv_faces, model.victim_embedding_input: target_embeddings,
                             model.batch_size: 10, model.phase_train: False}
                accur = model_eval(sess, x, y, preds, adv_faces, labels_batch,
                                 feed=feed_dict, args={'batch_size': args.lfw_batch_size})
  
                # Perform attack
                model.face_input.set_shape(np.shape(adv_faces))
                cw = CarliniWagnerL2(model, sess)
                cw_params = {'binary_search_steps': 1,
                             'max_iterations': 100,
                             'learning_rate': .2,
                             'batch_size': args.lfw_batch_size,
                             'initial_const': 10}
                adv_x = cw.generate(model.face_input, feed_dict, **cw_params)
                adv = sess.run(adv_x, feed_dict)
  
                feed_dict_adv = {model.face_input: adv, model.victim_embedding_input: target_embeddings,
                             model.batch_size: 10, model.phase_train: False}
                accur = model_eval(sess, x, y, preds, adv, labels_batch,
                                   feed=feed_dict, args={'batch_size': args.lfw_batch_size})
                """

                    # Evaluate accuracy
                    real_labels = np.append(real_labels, np.argmax(real_labels_batch, axis=-1))
                    adversarial_labels = np.append(adversarial_labels, np.argmax(adversarial_labels_batch, axis=-1))

    # Compute accuracy
    evaluate(labels_evaluated, real_labels, adversarial_labels)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--lfw_dir', type=str, help='Path to the data directory containing aligned LFW data.',
                        default='/data/jihyec/data/lfw/lfw-deepfunneled-mtcnnpy_dim160')
    parser.add_argument('--lfw_file_ext', type=str,
                        help='The file extension for the LFW dataset.', default='jpg', choices=['jpg', 'png'])
    parser.add_argument('--lfw_batch_size', type=int,
                        help='Number of images to process in a batch in the LFW test set.', default=10)
    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.',
                        default='../data/lfw-view2_pairs.txt')
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=1)
    parser.add_argument('--model_path', type=str, help='Type of non-attribute based model to be evaluated.',
                        default='facenet')
    parser.add_argument('--attack_type', type=str, help='Type of the attack method: FGSM, CW or random', default='CW')
    parser.add_argument('--eps', type=float, help='FGSM or random: Norm of adversarial perturbation.', default=0.9)
    parser.add_argument('--init_c', type=float, help='CW: Initial tradeoff-constant to use to tune the relative importance of size of the perturbation confidence of classification',
                        default=10)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

