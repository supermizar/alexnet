from network import Network
from fc_layer import FcLayer
from conv_layer import ConvLayer
from conv_layer_hidden import ConvLayerHidden
from activator import ReluActivator as Relu
from max_pooling_layer import MaxPoolingLayer
from softmax_layer import SoftmaxLayer
from lrn_layer import LrnLayer
from dropout_layer import DropoutLayer
import numpy as np

if __name__ == '__main__':
    fake_image = np.random.uniform(0, 255, [3, 50, 50])
    fake_label = np.zeros([10])
    fake_label[5] = 1
    net = Network()

    ConvLayer(net, 50, 50, 3, 11, 11, 48, 2, 4, Relu(), 0.05)
    DropoutLayer(net, dropout_prob=0.4)
    MaxPoolingLayer(net, 3, 3, 2)
    ConvLayerHidden(net, 3, 3, 24, 2, 3, Relu(), 0.05)
    LrnLayer(net, 2, 0.0001, 5, 0.75)
    MaxPoolingLayer(net, 2, 2, 1)
    FcLayer(net, 5, Relu())
    SoftmaxLayer(net, 10)

    net.train_one_sample(fake_label, fake_image, 0.1)
