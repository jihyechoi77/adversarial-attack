import argparse
import sys
from keras.models import load_model


def main(args):
    # load model
    if args.custom_loss:
        from train_utils import binary_crossentropy_custom
        import keras.activations
        keras.activations.custom_activation = binary_crossentropy_custom
        model = load_model(args.model_path + '.h5',
                           custom_objects={'binary_crossentropy_custom': binary_crossentropy_custom})
    else:
        model = load_model(args.model_path + '.h5')





def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('lfw_dir', type=str, help='Path to the data directory containing aligned LFW data.',
                        default='/home/jihyec/data/lfw/lfw-deepfunneled-mtcnnpy_dim160')
    parser.add_argument('model_path', type=str, help='Path to the trained attribute classifier.',
                        default='/home/jihyec/adversarial-ml/vgg_keras_tf/model/VGG16-dim160-attrClassifier.h5')
    parser.add_argument('feat_eng', type=bool, help='Whether to perform feature engineering with attribute vectors')
    parser.add_argument('custom_loss', type=bool, help='Whether the attrClassifier trained with custom loss')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))