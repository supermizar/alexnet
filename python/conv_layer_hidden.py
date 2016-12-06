from conv_layer import ConvLayer
import numpy as np


class ConvLayerHidden(ConvLayer):
    def __init__(self, network, filter_width,
                 filter_height, filter_number,
                 zero_padding, stride, activator,
                 learning_rate, dropout_prob=0.0):
        self.dropout_prob = dropout_prob
        upstream_layer = network.layers[-1]
        input_width = upstream_layer.get_output().shape[1]
        input_height = upstream_layer.get_output().shape[2]
        channel_number = upstream_layer.get_output().shape[0]
        ConvLayer.__init__(self, network, input_width, input_height,
                           channel_number, filter_width,
                           filter_height, filter_number,
                           zero_padding, stride, activator,
                           learning_rate)

    def forward(self, input_array):
        if self.dropout_prob > 0.0:
            retain_prob = 1. - self.dropout_prob
            dropout_array = np.random.binomial(1, retain_prob, size=input_array.shape)
            input_array *= dropout_array
            input_array /= retain_prob
        ConvLayer.forward(self, input_array)
