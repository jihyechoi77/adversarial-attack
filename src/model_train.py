import tensorflow as tf
import keras
import sys
import os
import numpy as np
from scipy.io import loadmat
import argparse
import random
import facenet

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import backend as K


from itertools import chain, repeat, cycle
def grouper(n, iterable, padvalue=None):
    g = cycle(zip(*[chain(iterable, repeat(padvalue, n-1))]*n))
    for batch in g:
        yield list(filter(None, batch))
 
 
def multilabel_flow(path_to_data, idg, bs, target_size, train_or_valid, label_all):
    gen = idg.flow_from_directory(path_to_data, batch_size=bs, target_size=target_size, classes=[train_or_valid], shuffle=False)
    names_generator = grouper(bs, gen.filenames)
    for (X_batch, _), names in zip(gen, names_generator):
        names = [n.split('/')[-1].replace('.jpg','') for n in names]
        idx = []
        print(names[0:3])
        y_batch = label_all[idx]

        yield X_batch, y_batch
 

def get_BatchGenerator(args):
# prepare python BatchGenerator
# reference: https://keras.io/preprocessing/image/
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

#    train_generator = train_datagen.flow_from_directory(os.path.join(args.data_dir,'train'),
#        target_size=(args.data_dim, args.data_dim),
#        batch_size=args.batch_size,
#        class_mode=None)
#
#    validation_generator = test_datagen.flow_from_directory(os.path.join(args.data_dir, 'validation'),
#        target_size=(args.data_dim, args.data_dim),
#        batch_size=args.batch_size,
#        class_mode=None) # class_mode: one of "categorical", "binary", "sparse", "input", "other" or None


    # custom batch generator
    # reference: 
    train_generator = multilabel_flow(args.data_dir,
                            train_datagen, 
                            bs=args.batch_size,
                            target_size=(args.data_dim,args.data_dim), 
                            train_or_valid='train',
                            label_all=args.label_all)
 
    validation_generator = multilabel_flow(args.data_dir,
                            test_datagen, 
                            bs=args.batch_size,
                            target_size=(args.data_dim,args.data_dim), 
                            train_or_valid='validation',
                            label_all=args.label_all)

    test_generator = test_datagen.flow_from_directory(args.data_dir,
                        target_size=(args.data_dim,args.data_dim),
                        classes=['test'],
                        shuffle=False)

    return train_generator, validation_generator, test_generator


def load_PretrainedModel(model_type, input_dim, output_dim):
# reference: https://keras.io/applications/#inceptionv3
    from keras.layers import Flatten, Dense, GlobalAveragePooling2D   

    # create the base pre-trained model
    if model_type=='VGG16' :
        from keras.applications.vgg16 import VGG16
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(input_dim,input_dim,3))
        feat_dim = 4096
    elif model_type=='InceptionResNetV2':
        from keras.applications.InceptionResNetV2 import InceptionResNetV2
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
    model.compile(optimizer='rmsprop', loss=args.loss)

    # train the model on the new data for a few epochs
    model.fit_generator(train_generator,
        steps_per_epoch=2000, epochs=args.epoch,
        validation_data=validation_generator, validation_steps=200)


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
        steps_per_epoch=2000, epochs=args.epoch,
        validation_data=validation_generator, validation_steps=800)


    return model




def main(args):

    np.random.seed(seed=args.seed)
    random.seed(args.seed)

    # prepare pretrained model
    base_model, model = load_PretrainedModel(args.model_type, args.data_dim, output_dim=40)


    # load Nx40 labels from mat file
    # mat file is located in data_dir
    loaded = loadmat(os.path.join(args.data_dir,'label_all.mat'))
    args.label_all = loaded['label']   

    # prepare python batch generators for training and validation
    train_generator, validation_generator, test_generator = get_BatchGenerator(args)

    # train
    trained_model = model_train(base_model, model, train_generator, validation_generator, args)
    trained_model.save("./model/%s-dim%d-attrClassifier.h5" % (model_type, input_dim))




def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, help='Path to the data directory containing aligned face images.', default='/home/jihyec/data/celeba/img_align_celeba-dim160')
    parser.add_argument('--model_type', type=str, help='Possible model types are VGG16, InceptionResNetV2.', default='VGG16')
    parser.add_argument('--seed', type=int, help='Random seed.', default=666)
    parser.add_argument('--epoch', type=int, help='Number of epochs.', default=100)  
    parser.add_argument('--batch_size', type=int, help='Batch size.', default=64)
    parser.add_argument('--data_dim', type=int, help='Dimension of images.', default=160)
    parser.add_argument('--loss', type=str, help='Training loss: either binary_crossentropy or .', default='binary_crossentropy')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
