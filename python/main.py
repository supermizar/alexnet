from network import Network
import fc_layer
import conv_layer

if __name__ == '__main__':
    net = Network()
    net = conv_layer.ConvLayer(net, 224, 1, 1, 4, 1, 4, 1, 3, conv_layer.ReluActivator, 0.05)
    net = fc_layer.Layer(net, 1, 10)
