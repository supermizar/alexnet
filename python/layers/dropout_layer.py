import numpy as np


class DropoutLayer(object):
    def __init__(self, network, dropout_prob=0.5):
        self.dropout_prob = dropout_prob
        input_array = network.layers[-1].get_output()
        self.output_array = np.zeros(input_array.shape)
        network.append_layer(self)

    def forward(self, input_array, training=False):
        if training:
            retain_prob = 1. - self.dropout_prob
            self.dropout_array = np.random.binomial(1, retain_prob, size=input_array.shape)
            self.output_array = input_array * self.dropout_array / retain_prob
        else:
            self.output_array = input_array / self.dropout_prob

    def get_output(self):
        return self.output_array

    def calc_layer_delta(self, downstream_layer):
        self.delta_array = downstream_layer.get_transformed_delta() * self.dropout_array

    def update_weight(self, rate):
        pass

    def get_transformed_delta(self):
        return self.delta_array