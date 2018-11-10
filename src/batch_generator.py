import keras
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os


class DataGenerator(keras.utils.Sequence):
    # Generates batch data for Keras
    # reference: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

    def __init__(self, list_IDs, labels,
                 # data_dir='/home/jihyec/data/celeba/img_align_celeba-dim160',
                 batch_size=100, dim=(160,160), n_channels=3,
                 n_classes=10, shuffle=True, is_multilab=True):
        # Initialization
        # self.data_dir = data_dir
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.is_multilab = is_multilab # is it multi-label classification task?
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        if self.is_multilab:
            y = np.empty((self.batch_size, 40), dtype=int)
        else:
            y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # X[i,] = np.load('data/' + ID + '.npy')  # load images in npy format
            X[i, ] = img_to_array(load_img(ID)) / 255

            # Store class
            img_num = int(os.path.split(ID)[-1].split('.')[0])  # extract image number from filename
            y[i] = np.transpose(self.labels[img_num-1])

        if self.is_multilab:
            return X, y
        else:
            return X, keras.utils.to_categorical(y, num_classes=self.n_classes)




"""
from itertools import chain, repeat, cycle
from keras.preprocessing.image import ImageDataGenerator
def grouper(n, iterable, padvalue=None):
    g = cycle(zip(*[chain(iterable, repeat(padvalue, n - 1))] * n))
    for batch in g:
        yield list(filter(None, batch))


def multilabel_flow(path_to_data, idg, bs, target_size, train_or_valid, label_all):
    gen = idg.flow_from_directory(path_to_data, batch_size=bs, target_size=target_size, classes=[train_or_valid],
                                  shuffle=True)
    names_generator = grouper(bs, gen.filenames)
    for (X_batch, _), names in zip(gen, names_generator):
        bs_idx = [n.split('/')[-1].replace('.jpg', '') for n in names]
        bs_idx = [x - 1 for x in bs_idx]
        y_batch = label_all[bs_idx - 1]

        yield X_batch, y_batch


def get_BatchGenerator(args):
    # prepare python BatchGenerator
    # reference: https://keras.io/preprocessing/image/
    #    train_datagen = ImageDataGenerator(
    #        rescale=1./255,
    #        shear_range=0.1,
    #        zoom_range=0.1,
    #        horizontal_flip=True)
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

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
    train_generator = multilabel_flow(args.data_dir, train_datagen,
                                      bs=args.batch_size,
                                      target_size=(args.data_dim, args.data_dim),
                                      train_or_valid='train',
                                      label_all=args.label_all)

    validation_generator = multilabel_flow(args.data_dir, test_datagen,
                                           bs=args.batch_size,
                                           target_size=(args.data_dim, args.data_dim),
                                           train_or_valid='validation',
                                           label_all=args.label_all)

    test_generator = test_datagen.flow_from_directory(args.data_dir,
                                                      target_size=(args.data_dim, args.data_dim),
                                                      classes=['test'],
                                                      shuffle=False)

    return train_generator, validation_generator, test_generator
"""