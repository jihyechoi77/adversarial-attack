
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
