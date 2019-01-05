from FaceRecognitionNets import VGGNet
import scipy.misc
import numpy as np
from skimage.transform import resize


# weights_path = '/data/jihyec/vgg_keras_tf/dnns-in-python/weights/vgg10-recognition-nn-raw-weights.mat'
weights_path = '/data/jihyec/vgg_keras_tf/dnns-in-python/weights/vgg-Male-weights.mat'
net = VGGNet(weights_path)
im = scipy.misc.imread('/data/jihyec/vgg_keras_tf/dnns-in-python/dnns-in-python/data/demo-vgg.png')
im = resize(im, (80, 80))
# scipy.misc.imsave('/data/jihyec/vgg_keras_tf/dnns-in-python/resized_demo.jpg',im)
im = np.expand_dims(im, 0)
output = net.predict(im)
print(np.squeeze(output))

