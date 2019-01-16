# Original file: https://github.com/tensorflow/cleverhans/blob/master/examples/facenet_adversarial_faces/facenet_fgsm.py

from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
import argparse
import math
from cleverhans.model import Model
from cleverhans.attacks import FastGradientMethod
import facenet
import lfw


def load_testset(args, size):
  # Load images paths and labels
  pairs = lfw.read_pairs(args.lfw_pairs)
  paths, labels = lfw.get_paths(args.lfw_dir, pairs)
  image_size = 160

  # Random choice
  permutation = np.random.choice(len(labels), size, replace=False)
  paths_batch_1 = []
  paths_batch_2 = []

  for index in permutation:
    paths_batch_1.append(paths[index * 2])
    paths_batch_2.append(paths[index * 2 + 1])

  labels = np.asarray(labels)[permutation]
  paths_batch_1 = np.asarray(paths_batch_1)
  paths_batch_2 = np.asarray(paths_batch_2)

  # Load images
  faces1 = facenet.load_data(paths_batch_1, False, False, image_size)
  faces2 = facenet.load_data(paths_batch_2, False, False, image_size)

  # Change pixel values to 0 to 1 values
  min_pixel = min(np.min(faces1), np.min(faces2))
  max_pixel = max(np.max(faces1), np.max(faces2))
  faces1 = (faces1 - min_pixel) / (max_pixel - min_pixel)
  faces2 = (faces2 - min_pixel) / (max_pixel - min_pixel)

  # Convert labels to one-hot vectors
  onehot_labels = []
  for index in range(len(labels)):
    if labels[index]:
      onehot_labels.append([1, 0])
    else:
      onehot_labels.append([0, 1])

  return faces1, faces2, np.array(onehot_labels)


def load_images(path1, path2):

    # Load images
    faces1 = facenet.load_data(path1, False, False, image_size=160)
    faces2 = facenet.load_data(path2, False, False, image_size=160)

    # Change pixel values to 0 to 1 values
    min_pixel = min(np.min(faces1), np.min(faces2))
    max_pixel = max(np.max(faces1), np.max(faces2))
    faces1 = (faces1 - min_pixel) / (max_pixel - min_pixel)
    faces2 = (faces2 - min_pixel) / (max_pixel - min_pixel)

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

  def convert_to_classifier(self):
    # Create victim_embedding placeholder
    self.victim_embedding_input = tf.placeholder(
        tf.float32,
        shape=(None, 512)) # 128

    # Squared Euclidean Distance between embeddings
    distance = tf.reduce_sum(
        tf.square(self.embedding_output - self.victim_embedding_input),
        axis=1)
    print(distance)

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
    self.layer_names.append('logits') # 'probs'


  def fprop(self, x, set_ref=False):
    # print(dict(zip(self.layer_names, self.layers)))
    return dict(zip(self.layer_names, self.layers))


def run_attack(sess, model, target_faces, adv_faces, adv_x):
    graph = tf.get_default_graph()
    phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
    batch_size = graph.get_tensor_by_name("batch_size:0")

    """
    # test
    import scipy.misc
    scipy.misc.imsave('./target.jpg', target_faces[1])
    scipy.misc.imsave('./adv.jpg', adv_faces[1])
    """

    # Create target embeddings using Facenet itself
    feed_dict = {model.face_input: target_faces, phase_train_placeholder: False}
    target_embeddings = sess.run(model.embedding_output, feed_dict=feed_dict)

    # Run attack
    steps = 1
    adv = adv_faces
    for i in range(steps):
        print("FGSM step " + str(i + 1))
        feed_dict = {model.face_input: adv,
                     model.victim_embedding_input: target_embeddings,
                     phase_train_placeholder: False}
        adv = sess.run(adv_x, feed_dict=feed_dict)

   
    # Prediction with original images
    feed_dict = {model.face_input: adv_faces,
                 model.victim_embedding_input: target_embeddings,
                 phase_train_placeholder: False}
                 #batch_size: 100}
    benign_labels = sess.run(model.softmax_output, feed_dict=feed_dict)

    # Prediction with adversarial images
    feed_dict = {model.face_input: adv,
                 model.victim_embedding_input: target_embeddings,
                 phase_train_placeholder: False}
                 # batch_size: 100}
    adversarial_labels = sess.run(model.softmax_output, feed_dict=feed_dict)

    return benign_labels, adversarial_labels  


def main(args):
    with tf.Graph().as_default():
      with tf.Session() as sess:
        # Load model
        model = InceptionResnetV1Model()
        # Convert to classifier
        model.convert_to_classifier() 
  
   #     graph = tf.get_default_graph()
   #     phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
    #    batch_size = graph.get_tensor_by_name("batch_size:0")

        # Define FGSM for the model
        steps = 1
        # eps = args.eps
        alpha = args.eps / steps
        fgsm = FastGradientMethod(model)
        fgsm_params = {'eps': alpha,
                       'clip_min': 0.,
                       'clip_max': 1.}
        adv_x = fgsm.generate(model.face_input, **fgsm_params)

 
        # Load images paths and labels
        pairs = lfw.read_pairs(args.lfw_pairs)
        paths, true_issame = lfw.get_paths(args.lfw_dir, pairs)
        path1 = paths[0::2]
        path2 = paths[1::2]
        labels = 1*np.array(true_issame)  # (N, ) array of 0 of 1

        num_images = len(paths)
        num_batches = int(math.ceil(1.0*num_images / args.lfw_batch_size))
    
        for i in range(num_batches):
            start_idx = i*args.lfw_batch_size
            end_idx = min((i+1)*args.lfw_batch_size, num_images)

            path1_batch = path1[start_idx:end_idx]
            path2_batch = path2[start_idx:end_idx]
            labels_batch = labels[start_idx:end_idx]
 
            # Load pairs of faces and their labels in one-hot encoding
            # adv_faces, target_faces = load_images(path1_batch, path2_batch)
            adv_faces, target_faces, labels_batch = load_testset(args, 100)
            # Run attack
            real_labels, adversarial_labels = run_attack(sess, model, target_faces, adv_faces, adv_x)

            ###########
            # Evaluate accuracy
            labels_batch = np.argmax(labels_batch, axis=-1)
            real_labels = np.argmax(real_labels, axis=-1)
            adversarial_labels = np.argmax(adversarial_labels, axis=-1)

            accuracy = np.mean(labels_batch == real_labels)
            print('Accuracy: ' + str(accuracy * 100) + '%')

            same_faces_index = np.where((labels_batch == 1))
            different_faces_index = np.where((labels_batch == 0))

            accuracy = np.mean(labels_batch[same_faces_index[0]] == adversarial_labels[same_faces_index[0]])
            print('Accuracy against adversarial examples for same person faces (dodging): '
                  + str(accuracy * 100) + '%')

            accuracy = np.mean(labels_batch[different_faces_index[0]] == adversarial_labels[different_faces_index[0]])
            print('Accuracy against adversarial examples for different people faces (impersonation): '
                  + str(accuracy * 100) + '%')


"""

           # Create target embeddings using Facenet itself
            feed_dict = {model.face_input: target_faces, phase_train_placeholder: False}
            target_embeddings = sess.run(model.embedding_output, feed_dict=feed_dict)
        
            # Run attack
            steps = 1
            adv = adv_faces
            for i in range(steps):
                print("FGSM step " + str(i + 1))
                feed_dict = {model.face_input: adv,
                             model.victim_embedding_input: target_embeddings,
                             phase_train_placeholder: False}
                adv = sess.run(adv_x, feed_dict=feed_dict)
        
        
            # Prediction with original images
            feed_dict = {model.face_input: adv_faces,
                         model.victim_embedding_input: target_embeddings,
                         phase_train_placeholder: False}
                         #batch_size: 100}
            real_labels = sess.run(model.softmax_output, feed_dict=feed_dict)
        
            # Prediction with adversarial images
            feed_dict = {model.face_input: adv,
                         model.victim_embedding_input: target_embeddings,
                         phase_train_placeholder: False}
                         # batch_size: 100}
            adversarial_labels = sess.run(model.softmax_output, feed_dict=feed_dict)
"""


    

        # Load pairs of faces and their labels in one-hot encoding
#        faces1, faces2, labels = load_testset(args,150)
    
#        # Create victims' embeddings using Facenet itself
#        graph = tf.get_default_graph()
#        phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
#        feed_dict = {model.face_input: faces2,
#                     phase_train_placeholder: False}
#        victims_embeddings = sess.run(
#            model.embedding_output, feed_dict=feed_dict)
"""    
        # Define FGSM for the model
        steps = 1
        # eps = args.eps
        alpha = args.eps / steps
        fgsm = FastGradientMethod(model)
        fgsm_params = {'eps': alpha,
                       'clip_min': 0.,
                       'clip_max': 1.}
        adv_x = fgsm.generate(model.face_input, **fgsm_params)
"""    

        # Run FGSM
#        adv = faces1
#        for i in range(steps):
#          print("FGSM step " + str(i + 1))
#          feed_dict = {model.face_input: adv,
#                       model.victim_embedding_input: victims_embeddings,
#                       phase_train_placeholder: False}
#          adv = sess.run(adv_x, feed_dict=feed_dict)
    
        # Test accuracy of the model
        # batch_size = graph.get_tensor_by_name("batch_size:0")
    
#        feed_dict = {model.face_input: faces1,
#                     model.victim_embedding_input: victims_embeddings,
#                     phase_train_placeholder: False,
#                     batch_size: 64}
#        real_labels = sess.run(model.softmax_output, feed_dict=feed_dict)
    
#        accuracy = np.mean(
#            (np.argmax(labels, axis=-1)) == (np.argmax(real_labels, axis=-1))
#        )
#        print('Accuracy: ' + str(accuracy * 100) + '%')
    
        # Test accuracy against adversarial examples
#        feed_dict = {model.face_input: adv,
#                     model.victim_embedding_input: victims_embeddings,
#                     phase_train_placeholder: False,
#                     batch_size: 64}
#        adversarial_labels = sess.run(
#            model.softmax_output, feed_dict=feed_dict)
"""    
        same_faces_index = np.where((np.argmax(labels, axis=-1) == 0))
        different_faces_index = np.where((np.argmax(labels, axis=-1) == 1))
    
        accuracy = np.mean(
            (np.argmax(labels[same_faces_index], axis=-1)) ==
            (np.argmax(adversarial_labels[same_faces_index], axis=-1))
        )
        print('Accuracy against adversarial examples for '
              + 'same person faces (dodging): '
              + str(accuracy * 100)
              + '%')
    
        accuracy = np.mean(
            (np.argmax(labels[different_faces_index], axis=-1)) == (
                np.argmax(adversarial_labels[different_faces_index], axis=-1))
        )
        print('Accuracy against adversarial examples for '
              + 'different people faces (impersonation): '
              + str(accuracy * 100)
              + '%')
    
"""


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--lfw_dir', type=str, help='Path to the data directory containing aligned LFW data.',default='/data/jihyec/data/lfw/lfw-deepfunneled-mtcnnpy_dim160')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='jpg', choices=['jpg', 'png'])
    parser.add_argument('--lfw_batch_size', type=int, help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='../data/lfw-view2_pairs.txt')
    parser.add_argument('--lfw_nrof_folds', type=int, help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--model_path', type=str, help='Type of non-attribute based model to be evaluated.', default='facenet')
    parser.add_argument('--eps', type=float, help='Norm of adversarial perturbation.', default=0.01)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

