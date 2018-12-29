import argparse
import sys
import h5py
import tensorflow as tf
import keras
from keras.models import load_model
import facenet

def load_tf_save_h5(model_path, save_path):
    # model_path: path to meta file
    with tf.Session() as sess:
         # saver = tf.train.import_meta_graph(model_path)
         # saver.restore(sess, tf.train.latest_checkpoint('./'))
         facenet.load_model(model_path)

        # reference: https://geekyisawesome.blogspot.com/2018/06/savingloading-tensorflow-model-using.html
         with h5py.File(save_path, 'w') as f:
             for var in tf.trainable_variables():
                  key = var.name.replace('/', ' ')
                  value = sess.run(var)
                  f.create_dataset(key, data=value)


def main(args):
    # load_tf_save_h5(args.model_path, args.save_path)
    model = load_model(args.save_path)
    model.summary()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, help='Path to the data directory containing aligned face images.',
                            default='/home/jihyec/data/vgg/vgg-aligned-dim160-tightCrop-ready')
    parser.add_argument('--model_path', type=str, help='Path to pre-trained tf model.',
                            default='/home/jihyec/adversarial-ml/trained_models/inceptionresnetv1-msceleba/20180408-102900.pb')
    parser.add_argument('--save_path', type=str, help='Path to save the converted file in h5 format.',
                            default='/home/jihyec/adversarial-ml/vgg_keras_tf/model/inceptionresnetv1-msceleba-dim160-faceClassifier.h5')
    parser.add_argument('--seed', type=int, help='Random seed.', default=666)
    parser.add_argument('--epoch', type=int, help='Number of epochs.', default=10)
    parser.add_argument('--batch_size', type=int, help='Batch size.', default=200)
    parser.add_argument('--data_dim', type=int, help='Dimension of images.', default=160)
    parser.add_argument('--steps_per_epoch', type=int, default=500)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
