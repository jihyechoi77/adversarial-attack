from FaceRecognitionNets import OpenFaceNet
import scipy.misc
import numpy as np

# weights_path = '../openface/train-OpenFace/OpenFace10/openface10-recognition-nn-raw-weights.mat'
weights_path = '/data/jihyec/vgg_keras_tf/dnns-in-python/weights/openface10-recognition-nn-raw-weights.mat'
net = OpenFaceNet(weights_path)
im = scipy.misc.imread('./data/demo-openface.png')
im = np.expand_dims(im, 0)
output = net.predict(im)
#print('**\n', np.squeeze(output)[0])
#print('**\n', np.squeeze(output)[1])
#print('**\n', np.squeeze(output)[2])
print(np.squeeze(output))
print(output.shape)

