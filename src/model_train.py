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
from keras.optimizers import SGD
from batch_generator import DataGenerator


def load_PretrainedModel(model_type, input_dim, output_dim):
    # reference: https://keras.io/applications/#inceptionv3
    from keras.layers import Flatten, Dense, GlobalAveragePooling2D   

    # create the base pre-trained model
    if model_type == 'VGG16':
        from keras.applications.vgg16 import VGG16
        base_model = VGG16(weights=None, include_top=False, input_shape=(input_dim,input_dim,3))
        # base_model.load_weights('/home/jihyec/adversarial-ml/vgg_keras_tf/rcmalli_vggface_tf_notop_vgg16.h5') # when running on devbox
        base_model.load_weights('/data/jihyec/rcmalli_vggface_tf_notop_vgg16.h5') # when running on grey4
        feat_dim = 4096
    elif model_type == 'InceptionResNetV2':
        from keras.applications.inception_resnet_v2 import InceptionResNetV2
        base_model = InceptionResNetV2(include_top=False, weights='imagenet')
        feat_dim = 1024
    else:
        from keras.applications.inception_v3 import InceptionV3
        base_model = InceptionV3(include_top=False, weights='imagenet')
        feat_dim = 1024

    # add a global spatial average pooling layer
    x = base_model.output
    if model_type == 'VGG16':
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


def get_BatchGenerator(args):
    # prepare python batch generators for training and validation

    if 'celeba' in args.data_dir:
        img_list_train = facenet.get_image_paths(os.path.join(args.data_dir, 'train'))
        img_list_val = facenet.get_image_paths(os.path.join(args.data_dir, 'validation'))
        assert len(img_list_train) > 0, 'The training set should not be empty'
        partition = {'train': img_list_train, 'validation': img_list_val}  # IDs
        params = {'dim': (args.data_dim, args.data_dim),
                  'batch_size': args.batch_size, 'shuffle': True}

        train_generator = DataGenerator(partition['train'], args.label_all, **params)
        validation_generator = DataGenerator(partition['validation'], args.label_all, **params)

    else:
        from keras.preprocessing.image import ImageDataGenerator
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(os.path.join(args.data_dir, 'train'),
                                                            target_size=(args.data_dim, args.data_dim),
                                                            batch_size=args.batch_size, class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(os.path.join(args.data_dir, 'validation'),
                                                                target_size=(args.data_dim, args.data_dim),
                                                                batch_size=args.batch_size, class_mode='categorical')

    return train_generator, validation_generator


def model_train(base_model, model, train_generator, validation_generator, args):
    # reference: https://keras.io/applications/#vgg16

    # ----------------STEP 1)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
       layer.trainable = False
  
    # compile the model (should be done *after* setting layers to non-trainable)
    if args.loss=='binary_crossentropy_custom':
        from train_utils import binary_crossentropy_custom
        model.compile(optimizer='adam', loss=binary_crossentropy_custom, metrics=['binary_accuracy'])
    elif args.loss=='binary_crossentropy':
        model.compile(optimizer='adam', loss=args.loss, metrics=['binary_accuracy'])
    else:
        model.compile(optimizer='adam', loss=args.loss, metrics=['acc'])

    # train the model on the new data for a few epochs
    model.fit_generator(train_generator,
                        steps_per_epoch=args.steps_per_epoch, epochs=args.epoch,
                        validation_data=validation_generator, validation_steps=200,
                        use_multiprocessing=False)

    # ----------------STEP 2)
    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first n(=freeze_bottom) layers and unfreeze the rest:
    if args.model_type == 'VGG16':
        freeze_bottom = 6
    elif args.model_type == 'InceptionV3':
        freeze_bottom = 249
    else:
        freeze_bottom = 546

    for layer in model.layers[:freeze_bottom]:
        layer.trainable = False
    for layer in model.layers[freeze_bottom:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    sgd = SGD(lr=0.0002, momentum=0.9)
    if args.loss=='binary_crossentropy_custom':
        from train_utils import binary_crossentropy_custom
        model.compile(optimizer=sgd, loss=binary_crossentropy_custom, metrics=['binary_accuracy'])
    elif args.loss=='binary_crossentropy':
        model.compile(optimizer=sgd, loss=args.loss, metrics=['binary_accuracy'])
    else:
        model.compile(optimizer=sgd, loss=args.loss, metrics=['acc'])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit_generator(train_generator,
                        steps_per_epoch=1000, epochs=args.epoch+5,
                        validation_data=validation_generator, validation_steps=800)

    return model


def main(args):

    np.random.seed(seed=args.seed)
    random.seed(args.seed)

    # prepare pretrained model
    if args.train_multilab:
        # load Nx40 labels from mat file
        # mat file is located in data_dir
        loaded = loadmat('../data/label_all.mat')
        args.label_all = np.array(loaded['label'])

        base_model, model = load_PretrainedModel(args.model_type, args.data_dim, output_dim=40)
    else:
        base_model, model = load_PretrainedModel(args.model_type, args.data_dim, output_dim=2622)

    train_generator, validation_generator = get_BatchGenerator(args)

    # train
    trained_model = model_train(base_model, model, train_generator, validation_generator, args)
    trained_model.save("../model/%s.h5" % args.save_name)
    # TODO: add option for further training with trained models.


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, help='Path to the data directory containing aligned face images.', default='/data/jihyec/data/celeba/img_align_celeba-dim160') # img_align_celeba-dim160-all when running on devbox
    parser.add_argument('--model_type', type=str, help='Possible model types are VGG16, InceptionResNetV2, InceptionV3.', default='VGG16')
    parser.add_argument('--seed', type=int, help='Random seed.', default=666)
    parser.add_argument('--epoch', type=int, help='Number of epochs.', default=10)
    parser.add_argument('--batch_size', type=int, help='Batch size.', default=300)
    parser.add_argument('--data_dim', type=int, help='Dimension of images.', default=160)
    parser.add_argument('--loss', type=str, help='Training loss: either binary_crossentropy or binary_crossentropy_custom.', default='binary_crossentropy')
    parser.add_argument('--save_name', type=str, help='Name to save the trained model.', default='VGG16-dim160-attrClassifier')
    parser.add_argument('--steps_per_epoch', type=int, default=500)
    parser.add_argument('--train_multilab', type=str2bool, default=True)
    return parser.parse_args(argv)


def str2bool(value):
    return value.lower == 'true'


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
