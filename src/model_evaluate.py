import sys
import os
import argparse
import numpy as np
from scipy.io import loadmat
from keras.models import load_model
from batch_generator import DataGenerator
import facenet
from eval_utils import evaluate_multilab


def get_label_from_filename(list_IDs, label_all):
    y = np.empty((len(list_IDs), 40), dtype=int)

    # Generate data
    for i, ID in enumerate(list_IDs):
        img_num = int(os.path.split(ID)[-1].split('.')[0])  # extract image number from filename
        y[i] = np.transpose(label_all[img_num - 1])

    return y


def main(args):

    # load trained model
    model_path = os.path.join(args.model_dir, args.model_name)
    model = load_model(model_path + '.h5')

    # load Nx40 labels from mat file
    # mat file is located in data_dir
    loaded = loadmat(os.path.join(args.data_dir, 'label_all.mat'))
    args.label_all = np.array(loaded['label'])

    # custom batch generator
    img_list = facenet.get_image_paths(args.data_dir)
    assert len(img_list) > 0, 'The training set should not be empty'
    params = {'dim': (args.data_dim, args.data_dim),
              'batch_size': args.batch_size, 'shuffle': True}

    test_generator = DataGenerator(img_list, args.label_all, **params)
    prediction = model.predict_generator(test_generator, use_multiprocessing=False, verbose=0)

    # compute label-wise performance
    truth = get_label_from_filename(img_list, args.label_all)
    evaluate_multilab(prediction, truth, save_name=model_path)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Path to the test data', default='/home/jihyec/data/celeba/img_align_celeba-dim160-all/test')
    parser.add_argument('--model_dir', type=str, help='Directory where the saved model is located.',
                        default='/home/jihyec/adversarial-ml/vgg_keras_tf/model')
    parser.add_argument('--model_name', type=str, help='Name of the saved model.', default='VGG16-dim160-attrClassifier')
    parser.add_argument('--batch_size', type=int, help='Test batch size.', default=400)
    parser.add_argument('--data_dim', type=int, help='Input data size.', default=160)
    return parser.parse_args(argv)


if __name__ == 'main':
    main(parse_arguments(sys.argv[1:]))
