from network import Network
from fc_layer import FcLayer
from conv_layer import ConvLayer
from conv_layer_hidden import ConvLayerHidden
from activator import ReluActivator as Relu
from max_pooling_layer import MaxPoolingLayer
from softmax_layer import SoftmaxLayer
from lrn_layer import LrnLayer
from dropout_layer import DropoutLayer
from scipy import misc
import numpy as np

if __name__ == '__main__':
    raccoon_image = misc.imread('../res/face.png').reshape([3,224,224])
    label = np.zeros([1000])
    label[0] = 1
    net = Network()

    ConvLayer(net, 224, 224, 3, 11, 11, 96, 2, 4, Relu(), 0.05)
    LrnLayer(net, 2, 0.0001, 5, 0.75)
    MaxPoolingLayer(net, 3, 3, 2)

    ConvLayerHidden(net, 5, 5, 256, 2, 1, Relu(), 0.05)
    LrnLayer(net, 2, 0.0001, 5, 0.75)
    MaxPoolingLayer(net, 3, 3, 2)

    ConvLayerHidden(net, 3, 3, 384, 1, 1, Relu(), 0.05)

    ConvLayerHidden(net, 3, 3, 384, 1, 1, Relu(), 0.05)

    ConvLayerHidden(net, 3, 3, 256, 1, 1, Relu(), 0.05)
    MaxPoolingLayer(net, 3, 3, 2)

    FcLayer(net, 2048, Relu())
    DropoutLayer(net, dropout_prob=0.5)

    FcLayer(net, 2048, Relu())
    DropoutLayer(net)

    SoftmaxLayer(net, 1000)

    net.train_one_sample(label, raccoon_image, 0.1)

    # fake_image = np.random.uniform(0, 255, [3, 25, 25])
    # fake_label = np.zeros([10])
    # fake_label[5] = 1
    # net = Network()
    #
    # ConvLayer(net, 25, 25, 3, 11, 11, 48, 2, 4, Relu(), 0.05)
    # LrnLayer(net, 2, 0.0001, 5, 0.75)
    # MaxPoolingLayer(net, 3, 3, 2)
    #
    # SoftmaxLayer(net, 10)
    #
    # net.train_one_sample(fake_label, fake_image, 0.1)
