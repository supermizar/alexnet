from network import Network
from fc_layer import FcLayer
from conv_layer import ConvLayer
from conv_layer import ReluActivator

if __name__ == '__main__':
    net = Network()
    ConvLayer(net, 224, 1, 1, 4, 1, 4, 1, 3, ReluActivator, 0.05)
    FcLayer(net, 1, 10)
    FcLayer(net, 1, 10)
