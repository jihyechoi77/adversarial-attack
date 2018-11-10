import tensorflow as tf
import keras
import sys
import os
import numpy as np
from scipy.io import loadmat
import argparse
import random
import facenet
from keras.models import Model
from batch_generator import DataGenerator


def load_PretrainedModel(model_type, input_dim, output_dim):
# reference: https://keras.io/applications/#inceptionv3
    from keras.layers import Flatten, Dense, GlobalAveragePooling2D   

    # create the base pre-trained model
    if model_type=='VGG16' :
        from keras.applications.vgg16 import VGG16
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(input_dim,input_dim,3))
        feat_dim = 4096
    elif model_type=='InceptionResNetV2':
        from keras.applications.inception_resnet_v2 import InceptionResNetV2
        base_model = InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000) 
    else:
        from keras.applications.inception_v3 import InceptionV3
        base_model = InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        feat_dim=1024 

    # add a global spatial average pooling layer
    x = base_model.output
    if model_type=='VGG16':
        x = Flatten(name='flatten')(x)
        x = Dense(feat_dim, activation='relu', name='fc1')(x)
        x = Dense(feat_dim, activation='relu', name='fc2')(x)
    else:
        x = GlobalAveragePooling2D()(x)
        x = Dense(feat_dim, activation='relu', name='fc')(x)
    # add a logistic layer
    predictions = Dense(output_dim, activation='sigmoid', name='predictions')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return base_model, model


def model_train(base_model, model, train_generator, validation_generator, args):
    # reference: https://keras.io/applications/#vgg16

    # ----------------1)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
       layer.trainable = False
  
    # compile the model (should be done *after* setting layers to non-trainable)
    if args.loss=='binary_crossentropy_custom':
        from train_utils import binary_crossentropy_custom
        model.compile(optimizer='adam', loss=binary_crossentropy_custom, metrics=['binary_accuracy'])
    else:
        model.compile(optimizer='adam', loss=args.loss, metrics=['binary_accuracy'])

    # train the model on the new data for a few epochs
    model.fit_generator(train_generator,
                        steps_per_epoch=500, epochs=args.epoch,
                        validation_data=validation_generator, validation_steps=200,
                        use_multiprocessing=False)

    # ----------------2)
    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first n(=freeze_bottom) layers and unfreeze the rest:
    freeze_bottom = 5
    for layer in model.layers[:freeze_bottom]:
        layer.trainable = False
    for layer in model.layers[freeze_bottom:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss=args.loss)

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit_generator(train_generator,
        steps_per_epoch=1000, epochs=args.epoch,
        validation_data=validation_generator, validation_steps=800)

    return model


def main(args):

    np.random.seed(seed=args.seed)
    random.seed(args.seed)

    # prepare pretrained model
    base_model, model = load_PretrainedModel(args.model_type, args.data_dim, output_dim=40)


    # load Nx40 labels from mat file
    # mat file is located in data_dir
    loaded = loadmat(os.path.join(args.data_dir, 'label_all.mat'))
    args.label_all = np.array(loaded['label'])

    # prepare python batch generators for training and validation
    # option 1)
    # train_generator, validation_generator, test_generator = get_BatchGenerator(args)

    # option 2)
    img_list_train = facenet.get_image_paths(os.path.join(args.data_dir,'train'))
    img_list_val = facenet.get_image_paths(os.path.join(args.data_dir, 'validation'))
    assert len(img_list_train) > 0, 'The training set should not be empty'
    partition = {'train': img_list_train, 'validation': img_list_val} # IDs
    params = {'dim': (args.data_dim, args.data_dim),
              'batch_size': args.batch_size, 'shuffle': True}

    train_generator = DataGenerator(partition['train'], args.label_all, **params)
    validation_generator = DataGenerator(partition['validation'], args.label_all, **params)

    # train
    trained_model = model_train(base_model, model, train_generator, validation_generator, args)
    trained_model.save("./model/%s.h5" % (args.save_name))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, help='Path to the data directory containing aligned face images.', default='/home/jihyec/data/celeba/img_align_celeba-dim160-all')
    parser.add_argument('--model_type', type=str, help='Possible model types are VGG16, InceptionResNetV2.', default='VGG16')
    parser.add_argument('--seed', type=int, help='Random seed.', default=666)
    parser.add_argument('--epoch', type=int, help='Number of epochs.', default=10)
    parser.add_argument('--batch_size', type=int, help='Batch size.', default=200)
    parser.add_argument('--data_dim', type=int, help='Dimension of images.', default=160)
    parser.add_argument('--loss', type=str, help='Training loss: either binary_crossentropy or binary_crossentropy_custom.', default='binary_crossentropy')
    parser.add_argument('--save_name', type=str, help='Name to save the trained model.', default='VGG16-dim160-attrClassifier')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
