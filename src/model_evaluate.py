from keras.models import load_model
import sys
import os
import argparse
import numpy as np
from scipy.io import loadmat
from batch_generator import DataGenerator
import facenet
from eval_utils import evaluate_multilab


def get_label_from_filename(list_IDs, label_all):
    y = np.empty((len(list_IDs), 40), dtype=int)

    # Generate data
    for i, ID in enumerate(list_IDs):
        img_idx = int(os.path.split(ID)[-1].split('.')[0])  # extract image index from filename
        y[i] = np.transpose(label_all[img_idx - 1])

    return y


def main(args):

    # load trained model
    model_path = os.path.join(args.model_dir, args.model_name)

    if args.loss == 'binary_crossentropy':
        model = load_model(model_path + '.h5')
    else:
        from train_utils import binary_crossentropy_custom
        import keras.activations
        keras.activations.custom_activation = binary_crossentropy_custom
        model = load_model(model_path + '.h5', custom_objects={'binary_crossentropy_custom': binary_crossentropy_custom})

    # load Nx40 labels from mat file
    # mat file is located under the data directory
    loaded = loadmat('../data/label_all.mat')
    args.label_all = np.array(loaded['label'])

    # custom batch generator
    img_list = facenet.get_image_paths(os.path.join(args.data_dir, 'test'))
    assert len(img_list) > 0, 'The training set should not be empty'
    params = {'dim': (args.data_dim, args.data_dim),
              'batch_size': args.batch_size, 'shuffle': False}

    test_generator = DataGenerator(img_list, args.label_all, **params)
    prediction = model.predict_generator(test_generator, use_multiprocessing=False, verbose=0)

    # compute label-wise performance
    truth = get_label_from_filename(img_list, args.label_all)
    evaluate_multilab(prediction, truth, save_name=model_path)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Path to the data', default='/data/jihyec/data/celeba/img_align_celeba-dim160')
    parser.add_argument('--model_dir', type=str, help='Directory where the saved model is located.',
                        default='/data/jihyec/adversarial-attack/model')
    parser.add_argument('--model_name', type=str, help='Name of the saved model.', default='VGG16-dim160-attrClassifier')
    parser.add_argument('--batch_size', type=int, help='Test batch size.', default=400)
    parser.add_argument('--data_dim', type=int, help='Input data size.', default=160)
    parser.add_argument('--loss', type=str,
                        help='Training loss: either binary_crossentropy or binary_crossentropy_custom.',
                        default='binary_crossentropy')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
