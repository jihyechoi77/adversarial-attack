import scipy.io
from keras.layers import Input, Conv2D, Activation, MaxPooling2D, AvgPool2D, BatchNormalization
from keras.layers import Flatten, Dense, Dropout, Lambda, Concatenate, Layer, ZeroPadding2D
from keras.initializers import Constant
import keras.backend as K
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils.conv_utils import convert_kernel
import numpy as np


class LRN2D(Layer):
    """
    This code is a legacy code from Keras for Local Response Normalization.
    URL: 
      https://github.com/fchollet/keras/blob/97174dd298cf4b5be459e79b0181a124650d9148/keras/layers/normalization.py#L66
    """

    def __init__(self, alpha=1e-4, k=1, beta=0.75, n=5):
        if n % 2 == 0:
            raise NotImplementedError("LRN2D only works with odd n. n provided: " + str(n))
        super(LRN2D, self).__init__()
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def get_output(self, train):
        X = self.get_input(train)
        b, ch, r, c = X.shape
        half_n = self.n // 2
        input_sqr = T.sqr(X)
        extra_channels = T.alloc(0., b, ch + 2*half_n, r, c)
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :], input_sqr)
        scale = self.k
        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        return X / scale

    def get_config(self):
        return {"name": self.__class__.__name__,
                "alpha": self.alpha,
                "k": self.k,
                "beta": self.beta,
                "n": self.n}
    
class VGGNet:
    """
    Load a VGG DNN
    """

    # average_image = [129.1863, 104.7624, 93.5940]
    # average_image = [3.9446, 3.1539, 0.8388]
    average_image = [4.4372, 3.5636, 1.2989]

    def __init__(self, weights):
        """
        Initialize a VGGNet from given weights.
        Weights can be a path to a mat file containing  model's weights, or a
        list of numpy arrays containing weights.

        This function does not work with the Theano backend due to how Conv2D
        is handled.
        """
        if isinstance(weights, str):
            # load weights, when given a path
            loaded = scipy.io.loadmat(weights)
            weights = loaded['weights'][0]

        # construct the model and initialize weights from
        # the list of weights
        net_in = Input(shape=(80,80,3))  # (224,224,3)
        net_out = Activation('relu')\
                  (Conv2D(64, (3,3), padding='same', kernel_initializer=Constant(weights[0]), bias_initializer=Constant(np.squeeze(weights[1])))(net_in))
        net_out = MaxPooling2D(pool_size=(2,2), strides=2)\
                  (Activation('relu')\
                   (Conv2D(64, (3,3), padding='same', kernel_initializer=Constant(weights[2]), bias_initializer=Constant(np.squeeze(weights[3])))(net_out)))
        net_out = Activation('relu')\
                  (Conv2D(128, (3,3), padding='same', kernel_initializer=Constant(weights[4]), bias_initializer=Constant(np.squeeze(weights[5])))(net_out))
        net_out = MaxPooling2D(pool_size=(2,2), strides=2)\
                  (Activation('relu')\
                   (Conv2D(128, (3,3), padding='same', kernel_initializer=Constant(weights[6]), bias_initializer=Constant(np.squeeze(weights[7])))(net_out)))
        net_out = Activation('relu')\
                  (Conv2D(256, (3,3), padding='same', kernel_initializer=Constant(weights[8]), bias_initializer=Constant(np.squeeze(weights[9])))(net_out))
        net_out = Activation('relu')\
                  (Conv2D(256, (3,3), padding='same', kernel_initializer=Constant(weights[10]), bias_initializer=Constant(np.squeeze(weights[11])))(net_out))
        net_out = MaxPooling2D(pool_size=(2,2), strides=2)\
                  (Activation('relu')\
                   (Conv2D(256, (3,3), padding='same', kernel_initializer=Constant(weights[12]), bias_initializer=Constant(np.squeeze(weights[13])))(net_out)))
        net_out = Activation('relu')\
                  (Conv2D(512, (3,3), padding='same', kernel_initializer=Constant(weights[14]), bias_initializer=Constant(np.squeeze(weights[15])))(net_out))
        net_out = Activation('relu')\
                  (Conv2D(512, (3,3), padding='same', kernel_initializer=Constant(weights[16]), bias_initializer=Constant(np.squeeze(weights[17])))(net_out))
        net_out = MaxPooling2D(pool_size=(2,2), strides=2)\
                  (Activation('relu')\
                   (Conv2D(512, (3,3), padding='same', kernel_initializer=Constant(weights[18]), bias_initializer=Constant(np.squeeze(weights[19])))(net_out)))
        net_out = Activation('relu')\
                  (Conv2D(512, (3,3), padding='same', kernel_initializer=Constant(weights[20]), bias_initializer=Constant(np.squeeze(weights[21])))(net_out))
        net_out = Activation('relu')\
                  (Conv2D(512, (3,3), padding='same', kernel_initializer=Constant(weights[22]), bias_initializer=Constant(np.squeeze(weights[23])))(net_out))
        net_out = MaxPooling2D(pool_size=(2,2), strides=2)\
                  (Activation('relu')\
                   (Conv2D(512, (3,3), padding='same', kernel_initializer=Constant(weights[24]), bias_initializer=Constant(np.squeeze(weights[25])))(net_out)))
        net_out = Dropout(0.5)\
                  (Activation('relu')\
                   (Dense(4096, kernel_initializer=Constant(weights[26]), bias_initializer=Constant(np.squeeze(weights[27])))\
                    (Flatten()(net_out))))
        net_out = Dropout(0.5)\
                  (Activation('relu')\
                   (Dense(4096, kernel_initializer=Constant(weights[28]), bias_initializer=Constant(np.squeeze(weights[29])))(net_out)))
        net_out = Activation('softmax')\
                  (Dense(weights[30].shape[3], kernel_initializer=Constant(weights[30]), use_bias=False)(net_out))
        self.model = Model(net_in, net_out)
        # print model summary
        self.model.summary()

    #def predict(self, ims):
    def predict(self, ims):

        """
        Classify images
        """
        ims = ims.astype('float32')

        # normalizie by subtracting the average image
        ims[:,:,:,0] = ims[:,:,:,0] - VGGNet.average_image[0]
        ims[:,:,:,1] = ims[:,:,:,1] - VGGNet.average_image[1]
        ims[:,:,:,2] = ims[:,:,:,2] - VGGNet.average_image[2]
        # classify
        prediction = self.model.predict(ims, verbose=0)
        return prediction

def StridedConv2D(tensor_in, strides, padding, weights, biases):
    """
    Keras' 2D convolution was behaving in a nonsensical way when
    strides were > 1, so I simply hacked it to produce the output
    I'd expect
    """
    tensor_out = Conv2D(weights.shape[3], (weights.shape[0],weights.shape[1]), strides=1, padding=padding,\
                        kernel_initializer=Constant(weights), bias_initializer=Constant(np.squeeze(biases)))(tensor_in)
    shape = K.int_shape(tensor_out)
    tensor_out = Lambda(lambda x: x[:,0:shape[1]:strides,0:shape[2]:strides,:])(tensor_out)
    return tensor_out

def StridedMaxPooling2d(tensor_in, strides, pool_size, padding='same'):
    """
    Max pooling needed to be hacked too to work with striding
    """
    tensor_out = MaxPooling2D(pool_size=pool_size, strides=1, padding=padding)(tensor_in)
    shape = K.int_shape(tensor_out)
    tensor_out = Lambda(lambda x: x[:,0:shape[1]:strides,0:shape[2]:strides,:])(tensor_out)
    return tensor_out

def StridedAvgPooling2d(tensor_in, strides, pool_size, padding='same'):
    """
    Average pooling needed to be hacked too to work with striding
    """
    tensor_out = AvgPool2D(pool_size=pool_size, strides=1, padding=padding)(tensor_in)
    shape = K.int_shape(tensor_out)
    tensor_out = Lambda(lambda x: x[:,0:shape[1]:strides,0:shape[2]:strides,:])(tensor_out)
    return tensor_out

def InceptionBranchType1(net_in, weights, strides=None, padding=None):
    """
    Conv->Bnorm->Relu->Conv->Bnorm-Relu
    """
    if strides is None:
        net_out = Conv2D(weights[0].shape[3], (weights[0].shape[0],weights[0].shape[1]),\
                         padding='same',\
                         kernel_initializer=Constant(weights[0]), bias_initializer=Constant(np.squeeze(weights[1])))(net_in)
    else:
        net_out = StridedConv2D(net_in, strides[0], padding[0], weights[0], weights[1])
    net_out = BatchNormalization(gamma_initializer = Constant(np.squeeze(weights[2])),\
                                  beta_initializer = Constant(np.squeeze(weights[3])),\
                                  moving_mean_initializer=Constant(np.squeeze(weights[4][:,0])),\
                                  moving_variance_initializer=Constant(np.squeeze(weights[4][:,1]**2)))(net_out)
    net_out = Activation('relu')(net_out)
    if strides is None:
        net_out = Conv2D(weights[5].shape[3], (weights[5].shape[0],weights[5].shape[1]),\
                         padding='same',\
                         kernel_initializer=Constant(weights[5]), bias_initializer=Constant(np.squeeze(weights[6])))(net_out)
    else:
        net_out = StridedConv2D(net_out, strides[1], padding[1], weights[5], weights[6])
    net_out = BatchNormalization(gamma_initializer = Constant(np.squeeze(weights[7])),\
                                  beta_initializer = Constant(np.squeeze(weights[8])),\
                                  moving_mean_initializer=Constant(np.squeeze(weights[9][:,0])),\
                                  moving_variance_initializer=Constant(np.squeeze(weights[9][:,1]**2)))(net_out)
    net_out = Activation('relu')(net_out)
    return net_out

def InceptionBranchType2(net_in, weights):
    """
    MaxPool->Conv->Bnorm->Relu
    """
    net_out = StridedMaxPooling2d(net_in, 2, (3,3), 'valid')
    net_out = Conv2D(weights[0].shape[3], (weights[0].shape[0],weights[0].shape[1]),\
                      padding='same',\
                      kernel_initializer=Constant(weights[0]), bias_initializer=Constant(np.squeeze(weights[1])))(net_out)
    net_out = BatchNormalization(gamma_initializer = Constant(np.squeeze(weights[2])),\
                                  beta_initializer = Constant(np.squeeze(weights[3])),\
                                  moving_mean_initializer=Constant(np.squeeze(weights[4][:,0])),\
                                  moving_variance_initializer=Constant(np.squeeze(weights[4][:,1]**2)))(net_out)
    net_out = Activation('relu')(net_out)
    return net_out

def InceptionBranchType3(net_in, weights):
    """
    Conv->Bnorm->ReLu
    """
    net_out = Conv2D(weights[0].shape[3], (weights[0].shape[0],weights[0].shape[1]),\
                     padding='same',\
                     kernel_initializer=Constant(weights[0]), bias_initializer=Constant(np.squeeze(weights[1])))(net_in)
    net_out = BatchNormalization(gamma_initializer = Constant(np.squeeze(weights[2])),\
                                 beta_initializer = Constant(np.squeeze(weights[3])),\
                                 moving_mean_initializer=Constant(np.squeeze(weights[4][:,0])),\
                                 moving_variance_initializer=Constant(np.squeeze(weights[4][:,1]**2)))(net_out)
    net_out = Activation('relu')(net_out)
    return net_out

def InceptionBranchType4(net_in, alpha, weights):
    "Square->AvgPool->MulConst->Sqrt->Conv->Bnorm->Relu"
    net_out = Lambda(lambda x: x**2)(net_in)
    net_out = StridedAvgPooling2d(net_out, 3, (3,3), padding='valid')
    net_out = Lambda(lambda x: (x*alpha)**0.5)(net_out)
    net_out = InceptionBranchType3(net_out, weights)
    return net_out

class OpenFaceNet:
    """
    Load an OpenFace DNN
    """
    
    def __init__(self, weights):
        """
        Initialize an OpenFaceNet from given weights.
        Weights can be a path to a mat file containing  model's weights, or a
        list of numpy arrays containing weights.

        This function does not work with the Theano backend due to how Conv2D
        is handled.
        """
        if isinstance(weights, str):
            # load weights, when given a path
            weights = scipy.io.loadmat(weights)
            weights = weights['weights'][0]
        # construct the model and initialize weights from the list of weights
        #  - first part is sequential
        net_in = Input(shape=(96,96,3))
        net_in_normalized = Lambda(lambda x: x/255., name='input_normalization')(net_in) # normalize input to [0,1] range
        net_out = StridedConv2D(net_in_normalized, 2, 'same', weights[0], weights[1])
        net_out = BatchNormalization(gamma_initializer = Constant(np.squeeze(weights[2])),\
                                     beta_initializer = Constant(np.squeeze(weights[3])),\
                                     moving_mean_initializer=Constant(np.squeeze(weights[4][:,0])),\
                                     moving_variance_initializer=Constant(np.squeeze(weights[4][:,1]**2)))(net_out)
        net_out = Activation('relu')(net_out)
        net_out = StridedMaxPooling2d(net_out, 2, (3,3))
        net_out = LRN2D()(net_out)
        net_out = Conv2D(weights[5].shape[3], (weights[5].shape[0],weights[5].shape[1]),\
                         kernel_initializer=Constant(weights[5]), bias_initializer=Constant(np.squeeze(weights[6])))(net_out)
        net_out = BatchNormalization(gamma_initializer = Constant(np.squeeze(weights[7])),\
                                     beta_initializer = Constant(np.squeeze(weights[8])),\
                                     moving_mean_initializer=Constant(np.squeeze(weights[9][:,0])),\
                                     moving_variance_initializer=Constant(np.squeeze(weights[9][:,1]**2)))(net_out)
        net_out = Activation('relu')(net_out)
        net_out = Conv2D(weights[10].shape[3], (weights[10].shape[0],weights[10].shape[1]),\
                         padding='same',\
                         kernel_initializer=Constant(weights[10]), bias_initializer=Constant(np.squeeze(weights[11])))(net_out)
        net_out = BatchNormalization(gamma_initializer = Constant(np.squeeze(weights[12])),\
                                     beta_initializer = Constant(np.squeeze(weights[13])),\
                                     moving_mean_initializer=Constant(np.squeeze(weights[14][:,0])),\
                                     moving_variance_initializer=Constant(np.squeeze(weights[14][:,1]**2)))(net_out)
        net_out = LRN2D()\
                  (Activation('relu')(net_out))
        net_out = StridedMaxPooling2d(net_out, 2, (3,3))
        # - first inception module
        # -- first branch
        net_out1 = InceptionBranchType1(net_out, weights[15:15+10])
        # -- second branch
        net_out2 = InceptionBranchType1(net_out, weights[25:25+10])
        # -- third branch
        net_out3 = InceptionBranchType2(net_out, weights[35:35+5])
        net_out3 = ZeroPadding2D(padding=((3,4),(3,4)))(net_out3)
        # -- fourth branch
        net_out4 = InceptionBranchType3(net_out, weights[40:40+5])
        # - concat
        net_out = Concatenate(axis=3)([net_out1, net_out2, net_out3, net_out4])
        # - second inception module
        # -- first branch
        net_out1 = InceptionBranchType1(net_out, weights[45:45+10])
        # -- second branch
        net_out2 = InceptionBranchType1(net_out, weights[55:55+10])
        # -- third branch
        net_out3 = InceptionBranchType4(net_out, 9, weights[65:65+5])
        net_out3 = ZeroPadding2D(padding=((4,4),(4,4)))(net_out3)
        # -- fourth branch
        net_out4 = InceptionBranchType3(net_out, weights[70:70+5])
        # - concat
        net_out = Concatenate(axis=3)([net_out1, net_out2, net_out3, net_out4])
        # - third inception module
        # -- first branch
        net_out1 = InceptionBranchType1(net_out, weights[75:75+10],\
                                        strides=[1,2],\
                                        padding=['valid','same'])
        # -- second branch
        net_out2 = InceptionBranchType1(net_out, weights[85:85+10],\
                                        strides=[1,2],\
                                        padding=['valid','same'])
        # -- third branch
        net_out3 = StridedMaxPooling2d(net_out, 2, (3,3), padding='valid')
        net_out3 = ZeroPadding2D(padding=((0,1),(0,1)))(net_out3)
        # - concat
        net_out = Concatenate(axis=3)([net_out1, net_out2, net_out3])
        # - fourth inception module
        # -- first branch
        net_out1 = InceptionBranchType1(net_out, weights[95:95+10])
        # -- second branch
        net_out2 = InceptionBranchType1(net_out, weights[105:105+10])
        # -- third branch
        net_out3 = InceptionBranchType4(net_out, 9, weights[115:115+5])
        net_out3 = ZeroPadding2D(padding=((2,2),(2,2)))(net_out3)
        # -- fourth branch
        net_out4 = InceptionBranchType3(net_out, weights[120:120+5])
        # - concat
        net_out = Concatenate(axis=3)([net_out1, net_out2, net_out3, net_out4])
        # - fifth inception module
        # -- first branch
        net_out1 = InceptionBranchType1(net_out, weights[125:125+10],\
                                        strides=[1,2],\
                                        padding=['valid','same'])
        # -- second branch
        net_out2 = InceptionBranchType1(net_out, weights[135:135+10],\
                                        strides=[1,2],\
                                        padding=['valid','same'])
        # -- third branch
        net_out3 = StridedMaxPooling2d(net_out, 2, (3,3), padding='valid')
        net_out3 = ZeroPadding2D(padding=((0,1),(0,1)))(net_out3)
        # - concat
        net_out = Concatenate(axis=3)([net_out1, net_out2, net_out3])
        # - sixth inception module
        # -- first branch
        net_out1 = InceptionBranchType1(net_out, weights[145:145+10])
        # -- second branch
        net_out2 = InceptionBranchType4(net_out, 9, weights[155:155+5])
        net_out2 = ZeroPadding2D(padding=((1,1),(1,1)))(net_out2)
        # -- third branch
        net_out3 = InceptionBranchType3(net_out, weights[160:160+5])
        # - concat
        net_out = Concatenate(axis=3)([net_out1, net_out2, net_out3])
        # - seventh inception module
        # -- first branch
        net_out1 = InceptionBranchType1(net_out, weights[165:165+10])
        # -- second branch
        net_out2 = InceptionBranchType2(net_out, weights[175:175+5])
        net_out2 = ZeroPadding2D(padding=((1,1),(1,1)))(net_out2)
        # -- third branch
        net_out3 = InceptionBranchType3(net_out, weights[180:180+5])
        # - concat
        net_out = Concatenate(axis=3)([net_out1, net_out2, net_out3])
        # done with inception layers from now on it's all sequential
        net_out = AvgPool2D(pool_size=(3,3), padding='valid')(net_out)
        net_out = Flatten()(net_out)
        net_out = Dense(weights[185].shape[0],\
                        kernel_initializer=Constant(np.transpose(weights[185], (1,0))),\
                        bias_initializer=Constant(np.squeeze(weights[186])))(net_out)
        net_out = Lambda(lambda x: K.l2_normalize(x, axis=1))(net_out)
        weights[187] = np.squeeze(weights[187])
        weights[189] = np.squeeze(weights[189])
        net_out = Dense(weights[187].shape[1],\
                        kernel_initializer=Constant(weights[187]),\
                        bias_initializer=Constant(np.squeeze(weights[188])))(net_out)
        net_out = Activation('tanh')(net_out)
        net_out = Dense(weights[189].shape[1],\
                        kernel_initializer=Constant(weights[189]),\
                        bias_initializer=Constant(np.squeeze(weights[190])))(net_out)
        net_out = Activation('softmax')(net_out)
        # create model
        self.model = Model(net_in, net_out)
        # print model summary
        self.model.summary()

    def predict(self, ims):
        """
        Classify images
        """
        # cast
        ims = ims.astype('float32')
        # classify
        prediction = self.model.predict(ims, verbose=0)
        return prediction
