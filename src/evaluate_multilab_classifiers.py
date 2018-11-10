# to evaluate the performance of various multilabel classifiers on CelebA dataset
# they should be evaluated on the same constitution of test data from CelebA

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from load_data import load_data_celeba_multiple
from eval_utils import evaluate_multilab
from train_utils import convert_y_range
from models_ConvNets import VGG16


evaluate_size = 10  # evaluate with CelebA fold 1~N
input_dim = 140


def load_vgg_model():
    from models_ConvNets import VGG16
    model = VGG16(input_shape=(input_dim, input_dim, 3), nb_classes=40, last_activation='sigmoid')

    return model


def load_inception_model():
    from model_inceptionResnetV2 import InceptionResNetV2
    model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=[input_dim, input_dim, 3])

    # add classification block
    from keras.layers import GlobalAveragePooling2D, Dense
    from keras.models import Model
    x = GlobalAveragePooling2D(name='avg_pool')(model.layers[-1].output)
    x = Dense(40, activation='sigmoid', name='predictions')(x)
    model = Model(model.layers[0].input, x)

    return model


def evaluate_on_ith_celeba_data(ith_set, model):
    # Get CelebA dataset
    (_, _), (X_test, truth) = load_data_celeba_multiple(start_set=ith_set, end_set=ith_set, image_dim=input_dim)
    truth, _ = convert_y_range(truth, _)
    pred = model.predict(X_test)

    return truth, pred


def main(modelname):
    modelpath = "/data/jihyec/vgg_keras_tf/trained_models/%s_notop_on_celeba.h5" % modelname

    if 'VGG16' in modelname:
        model = load_vgg_model()
    else:
        model = load_inception_model()
    model.summary()
    model.load_weights(modelpath)
    print("Model loaded: %s_notop_on_celeba.h5" % modelname)

    truth = np.empty((0, 40), float)
    pred = np.empty((0, 40), float)
    for ith_set in range(1, evaluate_size+1):
        ith_truth, ith_pred = evaluate_on_ith_celeba_data(ith_set, model)

        pred = np.append(pred, ith_pred, axis=0)
        truth = np.append(truth, ith_truth, axis=0)

    evaluate_multilab(modelname, preds=pred, truth=truth)


if __name__ == '__main__':
    # modelname : VGG16s, VGG16_multilab, VGG16_multilab_using_customloss, InceptionResnetV2_multilab
    # main(modelname='VGG16_multilab')
    main(modelname='VGG16_multilab_using_customloss')
    main(modelname='InceptionResnetV2_multilab')
    main(modelname='InceptionResnetV2_multilab_using_customloss')
