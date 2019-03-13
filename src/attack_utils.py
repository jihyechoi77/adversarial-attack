import numpy as np
import scipy.misc


def evaluate(true_labels, real_labels, adversarial_labels):
    accuracy = np.mean(true_labels == real_labels)
    print('Accuracy: ' + str(accuracy * 100) + '%')

    same_faces_index = np.where((true_labels == 0))
    different_faces_index = np.where((true_labels == 1))

    accuracy = np.mean(true_labels[same_faces_index[0]] == adversarial_labels[same_faces_index[0]])
    print('Number of same person faces: ' + str(sum(same_faces_index[0])))
    print('Accuracy against adversarial examples for same person faces (dodging): '
          + str(accuracy * 100) + '%')

    accuracy = np.mean(true_labels[different_faces_index[0]] == adversarial_labels[different_faces_index[0]])
    print('Number of different people faces: ' + str(sum(different_faces_index[0])))
    print('Accuracy against adversarial examples for different people faces (impersonation): '
          + str(accuracy * 100) + '%')


def visualize_image(images, index, save_path):
    # images: numpy array of images
    # index: index to images to be saved excluding the file extension (ex. './adv')
    # save_path: path to save image

    for i in index:
        scipy.misc.imsave(save_path+i+'.png', images[i])
