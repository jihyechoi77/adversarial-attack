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
from cleverhans.model import Model
from cleverhans.attacks import FastGradientMethod, CarliniWagnerL2
from cleverhans.utils_tf import model_eval
from keras.utils import to_categorical
import facenet
import lfw


def load_images(path1, path2):

    # Load images
    faces1 = facenet.load_data(path1, False, False, image_size=160)
    faces2 = facenet.load_data(path2, False, False, image_size=160)

    # Change pixel values to 0 to 1 values
    min_pixel = min(np.min(faces1), np.min(faces2))
    max_pixel = max(np.max(faces1), np.max(faces2))
    faces1 = (faces1 - min_pixel) / (max_pixel - min_pixel) -0.5
    faces2 = (faces2 - min_pixel) / (max_pixel - min_pixel) - 0.5

    return faces1, faces2


class InceptionResnetV1Model(Model):
  model_path = "../model/tf/20180402-114759.pb"

  def __init__(self):
    super(InceptionResnetV1Model, self).__init__(scope='model')

    # Load Facenet CNN
    facenet.load_model(self.model_path)
    # Save input and output tensors references
    graph = tf.get_default_graph()
    self.face_input = graph.get_tensor_by_name("input:0")
    self.embedding_output = graph.get_tensor_by_name("embeddings:0")
    self.phase_train = graph.get_tensor_by_name("phase_train:0")
    self.batch_size = graph.get_tensor_by_name("batch_size:0")

    # get all tensor names in the graph
    # tensor_names = [t.name for op in graph.get_operations() for t in op.values()]

    # to use Carlini attack
    self.image_size = 160
    self.num_channels = 3
    self.num_labels = 2

  def convert_to_classifier(self):
    # Create victim_embedding placeholder
    self.victim_embedding_input = tf.placeholder(tf.float32, shape=(None, 512), name="victim_embedding_input") # 128

    # Squared Euclidean Distance between embeddings
    distance = tf.reduce_sum(
        tf.square(self.embedding_output - self.victim_embedding_input),
        axis=1)

    # Convert distance to a softmax vector
    # 0.99 out of 4 is the distance threshold for the Facenet CNN
    threshold = 0.99
    score = tf.where(
        distance > threshold,
        0.5 + ((distance - threshold) * 0.5) / (4.0 - threshold),
        0.5 * distance / threshold)
    reverse_score = 1.0 - score
    self.softmax_output = tf.transpose(tf.stack([reverse_score, score]))

    # Save softmax layer
    self.layer_names = []
    self.layers = []
    self.layers.append(self.softmax_output)
    self.layer_names.append('logits')  # 'probs'

  def fprop(self, x, set_ref=False):
    # print(dict(zip(self.layer_names, self.layers)))
    return dict(zip(self.layer_names, self.layers))
"""
  def predict(self, sess, data):
    # to use Carlini attack (see line 90 of l2_attack.py)
    # prediction BEFORE - SOFTMAX of the model
    feed_dict = {self.face_input: data}  # , phase_train_placeholder: False}
    return sess.run(self.softmax_output, feed_dict=feed_dict)
"""


def prepare_attack(sess, args, model, adv_input, target_embeddings):
    if args.attack_type == 'FGSM':
        # Define FGSM for the model
        steps = 1
        # eps = args.eps
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
        cw_params = {'binary_search_steps': 0,  # 1
                     'max_iterations': 0,  # 100
                     'learning_rate': .1, # .2
                     'batch_size': args.lfw_batch_size,
                     'initial_const': 10}  # initial_const: 10, The initial tradeoff-constant to use to tune the
                                           # relative importance of size of the perturbation confidence of classification
        # adv_x = cw.generate_np(adv_input, **cw_params)
        feed_dict = {model.face_input: adv_input, model.victim_embedding_input: target_embeddings,
                     model.batch_size: 10, model.phase_train: False}
        adv_x = cw.generate(model.face_input, feed_dict, **cw_params)

    return adv_x


def run_attack(sess, model, adv_x, adv_faces, feed_dict):

    """
    # test
    import scipy.misc
    scipy.misc.imsave('./target.jpg', target_faces[1])
    scipy.misc.imsave('./adv.jpg', adv_faces[1])
    """

    # Run attack
    steps = 1
    adv = adv_faces
    for i in range(steps):
        # print("FGSM step " + str(i + 1))
        adv = sess.run(adv_x, feed_dict=feed_dict)
#        adv = sess.run(adv_x)

    # Prediction with original images
    benign_labels = sess.run(model.softmax_output, feed_dict)

    # Prediction with adversarial images
    feed_dict[model.face_input] = adv
    adversarial_labels = sess.run(model.softmax_output, feed_dict)

    return benign_labels, adversarial_labels  


def evaluate(true_labels, real_labels, adversarial_labels):
    accuracy = np.mean(true_labels == real_labels)
    print('Accuracy: ' + str(accuracy * 100) + '%')

    same_faces_index = np.where((true_labels == 0))
    different_faces_index = np.where((true_labels == 1))

    accuracy = np.mean(true_labels[same_faces_index[0]] == adversarial_labels[same_faces_index[0]])
    print('Accuracy against adversarial examples for same person faces (dodging): '
          + str(accuracy * 100) + '%')

    accuracy = np.mean(true_labels[different_faces_index[0]] == adversarial_labels[different_faces_index[0]])
    print('Accuracy against adversarial examples for different people faces (impersonation): '
          + str(accuracy * 100) + '%')


def main(args):
    with tf.Graph().as_default():
      with tf.Session() as sess:

          # Load model
          model = InceptionResnetV1Model()
          # Convert to classifier
          model.convert_to_classifier()

          # Load images paths and labels
          pairs = lfw.read_pairs(args.lfw_pairs)
          paths, true_issame = lfw.get_paths(args.lfw_dir, pairs)
          path1_all = paths[0::2]
          path2_all = paths[1::2]
          labels_all = 1 - 1*np.array(true_issame)  # (N, ) array of 0 of 1, N = 6000 for LFW view2 dataset
                                                # NOTE! for this example, 0: identical, 1: different

          k_fold = KFold(n_splits=10, shuffle=False)  # 10-fold cross validation
          indices = np.arange(np.shape(path1_all)[0])
          real_labels = []
          adversarial_labels = []

          for ith_fold, (_, fold_idx) in enumerate(k_fold.split(indices)):
              if ith_fold == args.lfw_nrof_folds:
                  break

              path1 = [path1_all[i] for i in fold_idx]
              path2 = [path2_all[i] for i in fold_idx]
              # labels = labels_all[fold_idx]
              num_images = len(path1)
              num_batches = int(math.ceil(1.0 * num_images / args.lfw_batch_size))

              for i in range(num_batches):
                  start_idx = i*args.lfw_batch_size
                  end_idx = min((i+1)*args.lfw_batch_size, num_images)
  
                  path1_batch = path1[start_idx:end_idx]
                  path2_batch = path2[start_idx:end_idx]
                  # labels_batch = to_categorical(labels[start_idx:end_idx], num_classes=2)
   
                  # Load pairs of faces and their labels in one-hot encoding
                  adv_faces, target_faces = load_images(path1_batch, path2_batch)

                  # Create target embeddings using Facenet itself
                  feed_dict = {model.face_input: target_faces, model.phase_train: False}
                  target_embeddings = sess.run(model.embedding_output, feed_dict=feed_dict)

                  # Run attack
                  feed_dict = {model.face_input: adv_faces, model.victim_embedding_input: target_embeddings,
                               model.batch_size: 10, model.phase_train: False}
                  adv_x = prepare_attack(sess, args, model, adv_faces, target_embeddings)
                  real_labels_batch, adversarial_labels_batch = run_attack(sess, model, adv_x, adv_faces, feed_dict)

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
    labels_evaluated = labels_all[:fold_idx[0]]  # true labels that belong to 0~N folds
    evaluate(labels_evaluated, real_labels, adversarial_labels)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--lfw_dir', type=str, help='Path to the data directory containing aligned LFW data.',default='/data/jihyec/data/lfw/lfw-deepfunneled-mtcnnpy_dim160')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='jpg', choices=['jpg', 'png'])
    parser.add_argument('--lfw_batch_size', type=int, help='Number of images to process in a batch in the LFW test set.', default=200)
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='../data/lfw-view2_pairs.txt')
    parser.add_argument('--lfw_nrof_folds', type=int, help='Number of folds to use for cross validation. Mainly used for testing.', default=2)
    parser.add_argument('--model_path', type=str, help='Type of non-attribute based model to be evaluated.', default='facenet')
    parser.add_argument('--attack_type', type=str, help='Type of the attack method: FGSM or CW', default='CW')
    parser.add_argument('--eps', type=float, help='Norm of adversarial perturbation.', default=0.01)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

