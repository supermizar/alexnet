import numpy as np
from layers.conv_layer import ConvLayer
from layers.conv_layer_hidden import ConvLayerHidden
from layers.fc_layer import FcLayer
from layers.lrn_layer import LrnLayer
from layers.max_pooling_layer import MaxPoolingLayer
from layers.softmax_layer import SoftmaxLayer
from utils.sci_plotter import SciPlot
from scipy import misc

from python.layers.dropout_layer import DropoutLayer
from python.network.network import Network
from python.utils.activator import ReluActivator as Relu

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

    net.train_one_sample(label, raccoon_image, 1, showlog=False)
    sp = SciPlot('Predict time of each layers')
    sp.bar(net.indexes, net.predict_time)
    sp = SciPlot('Delta calculation time of each layers')
    sp.bar(net.indexes, net.delta_calc_time)
    sp = SciPlot('Weight update time of each layers')
    sp.bar(net.indexes, net.update_weight_time)

    # fake_image = np.random.uniform(0, 255, [3, 25, 25])
    # fake_label = np.zeros([10])
    # fake_label[5] = 1
    # net = Network()
    #
    # # sp = SciPlot('Curve of softmax output')
    # ConvLayer(net, 25, 25, 3, 11, 11, 48, 2, 4, Relu(), 0.05)
    # LrnLayer(net, 2, 0.0001, 5, 0.75)
    # MaxPoolingLayer(net, 3, 3, 2)
    #
    # FcLayer(net, 5, Relu())
    # DropoutLayer(net,dropout_prob=0.5)
    # SoftmaxLayer(net, 10)

    # sp.plot(net.predict(fake_image, training=False), desc='episode-' + str(0))
    # for i in range(0, 5):
    #     net.train_one_sample(fake_label, fake_image, 1)
    #     sp.plot(net.predict(fake_image, training=False), desc='episode-'+str(i+1))
    #
    # sp.show()

    # net.train_one_sample(fake_label, fake_image, 1, showlog=False)
    # sp = SciPlot('Predict time of each layers')
    # sp.bar(net.indexes, net.predict_time)
    # sp = SciPlot('Delta calculation time of each layers')
    # sp.bar(net.indexes, net.delta_calc_time)
    # sp = SciPlot('Weight update time of each layers')
    # sp.bar(net.indexes, net.update_weight_time)
