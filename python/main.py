from network import Network
from fc_layer import FcLayer
from conv_layer import ConvLayer
from conv_layer_hidden import ConvLayerHidden
from activator import ReluActivator as Relu
from max_pooling_layer import MaxPoolingLayer
import numpy as np

if __name__ == '__main__':
    fake_input = np.random.uniform(16,32,[3,50,50])
    net = Network()
    ConvLayer(net, 50, 50, 3, 11, 11, 48, 2, 4, Relu(), 0.05)
    MaxPoolingLayer(net, 3, 3, 2)
    ConvLayerHidden(net, 3, 3, 24, 2, 3, Relu(), 0.05)
    MaxPoolingLayer(net, 2, 2, 1)
    FcLayer(net, 5, Relu())
    FcLayer(net, 10, Relu())
    net.train_one_sample(np.ones([10]), fake_input, 0.1)
