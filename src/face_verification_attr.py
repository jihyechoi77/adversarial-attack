import argparse
import sys
import os
from keras.models import load_model
import lfw
from eval_utils import compute_attr_embedding


def train_verifier(train_data):

def main(args):
    # load attribute classifier
    if args.custom_loss:
        from train_utils import binary_crossentropy_custom
        import keras.activations
        keras.activations.custom_activation = binary_crossentropy_custom
        model = load_model(args.model_path + '.h5',
                           custom_objects={'binary_crossentropy_custom': binary_crossentropy_custom})
    else:
        model = load_model(args.model_path + '.h5')

    # load LFW view1 data to train verifier
    # read the file containing the pairs used for testing
    pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
    # get the paths for the corresponding images
    paths, true_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, 'jpg')
    # compute embeddings
    emb1 = compute_attr_embedding(model, path)
    emb2 = compute_attr_embedding(model, path)
    # perform feature engineering

    # train verifier with attribute features


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--lfw_dir', type=str, help='Path to the data directory containing aligned LFW data.',
                        default='/data/jihyec/data/lfw/lfw-deepfunneled-dim160')
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='/data/jihyec/data/lfw/pairs.txt')
    parser.add_argument('--model_path', type=str, help='Path to the trained attribute classifier.',
                        default='/data/jihyec/adversarial-attack/model/VGG16-dim160-attrClassifier.h5')
    parser.add_argument('--save_verifier', type=str, help='Path to save the trained face verifier.', default='/data/jihyec/adversarial-attack/model/VGG16-dim160-attr-verifier.h5')
    parser.add_argument('feat_eng', type=bool, help='Whether to perform feature engineering with attribute vectors')
    parser.add_argument('custom_loss', type=bool, help='Whether the attrClassifier trained with custom loss')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
