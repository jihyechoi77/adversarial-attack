# reference: https://github.com/nyoki-mtl/keras-facenet/blob/master/notebook/tf_to_keras.ipynb

import os
import re
import numpy as np
import tensorflow as tf

import sys
#sys.path.append('../code/')
import argparse
from inception_resnet_v1 import *

# regex for renaming the tensors to their corresponding Keras counterpart
re_repeat = re.compile(r'Repeat_[0-9_]*b')
re_block8 = re.compile(r'Block8_[A-Za-z]')

def get_filename(key):
    filename = str(key)
    filename = filename.replace('/', '_')
    filename = filename.replace('InceptionResnetV1_', '')

    # remove "Repeat" scope from filename
    filename = re_repeat.sub('B', filename)

    if re_block8.match(filename):
        # the last block8 has different name with the previous 5 occurrences
        filename = filename.replace('Block8', 'Block8_6')

    # from TF to Keras naming
    filename = filename.replace('_weights', '_kernel')
    filename = filename.replace('_biases', '_bias')

    return filename + '.npy'


def extract_tensors_from_ckpt_file(filename, output_folder):
    reader = tf.train.NewCheckpointReader(filename)

    for key in reader.get_variable_to_shape_map():
        # not saving the following tensors
        if key == 'global_step':
            continue
        if 'AuxLogit' in key:
            continue

        # convert tensor name into the corresponding Keras layer weight name and save
        path = os.path.join(output_folder, get_filename(key))
        arr = reader.get_tensor(key)
        np.save(path, arr)


def load_npy_weights(model, npy_weights_dir):
    print('Loading numpy weights from', npy_weights_dir)
    for layer in model.layers:
        if layer.weights:
            weights = []
            for w in layer.weights:
                weight_name = os.path.basename(w.name).replace(':0', '')
                weight_file = layer.name + '_' + weight_name + '.npy'
                weight_arr = np.load(os.path.join(npy_weights_dir, weight_file))
                weights.append(weight_arr)
            layer.set_weights(weights)

    return model


def main(args):

    npy_weights_dir = os.path.join(args.tf_model_dir,args.tf_modelname.split('.')[0]+'_weights-npy')
    if not os.path.exists(npy_weights_dir):
        os.makedirs(npy_weights_dir)

    extract_tensors_from_ckpt_file(os.path.join(args.tf_model_dir,args.tf_modelname), npy_weights_dir)
    
    model = InceptionResNetV1()
    # model.summary()

    model = load_npy_weights(model, npy_weights_dir)

    print('Saving model...')
    model.save(os.path.join(args.keras_model_dir, args.tf_modelname.split('.')[0]+'.h5'))



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--tf_model_dir', type=str, help='Path to tf model', default='/data/jihyec/adversarial-attack/model/tf')
    parser.add_argument('--tf_modelname', type=str, help='ckpt file name of tf model', default='model-20170512-110547.ckpt-250000')
    parser.add_argument('--keras_model_dir', type=str, help='Path to save the generated keras model', default='/data/jihyec/adversarial-attack/model')


    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
