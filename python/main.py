from network import Network
from fc_layer import FcLayer
from conv_layer import ConvLayer
from activator import ReluActivator as Relu
import numpy as np

if __name__ == '__main__':
    fake_input = np.random.uniform(16,32,[3,224,224])
    net = Network()
    ConvLayer(net, 224, 224, 3, 11, 11, 48, 2, 4, Relu(), 0.05)
    FcLayer(net, 5, Relu())
    FcLayer(net, 10, Relu())
    net.train_one_sample(np.ones([10]), fake_input, 0.1)
