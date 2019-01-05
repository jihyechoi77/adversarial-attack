from keras.layers import Conv2D, Input
from keras.initializers import Constant
from keras.models import Model
import numpy as np

x = Input((3,3,1))
w = np.asarray([[1,-2,1],[-1,2,3],[-1,-2,-3]], dtype='float32')
print(w[0])
b = np.asarray([0.], dtype='float32')
print(b)
y = Conv2D(1, (3,3), kernel_initializer=Constant(w), bias_initializer=Constant(b), padding='same', strides=(2,2))(x)
model = Model(x,y)
model.summary()

input_arr = np.asarray( [[1,2,3],[4,5,6],[7,8,9]], dtype='float32')
input_arr = np.expand_dims(input_arr, axis=0)
input_arr = np.expand_dims(input_arr, axis=4)
output = model.predict(input_arr)
print(np.squeeze(output))
print(output.shape)
