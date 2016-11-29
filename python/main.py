from network import Network
from fc_layer import FcLayer
from conv_layer import ConvLayer
from conv_layer import ReluActivator
import numpy as np

if __name__ == '__main__':
    fake_input = np.ones([3,224,224])
    net = Network()
    ConvLayer(net, 224, 224, 3, 11, 11, 48, 2, 4, ReluActivator(), 0.05)
    output = net.predict(fake_input)
