import numpy as np


class LrnLayer(object):
    def __init__(self, network, k, alpha, n, beta):
        self.input_array = network.layers[-1].get_output()
        self.k = k
        self.alpha = alpha
        self.n = n
        self.beta = beta
        self.output_array = np.zeros(self.input_array.shape)
        network.append_layer(self)

    def forward(self, input_array):
        self.divisor_matrix = np.zeros(input_array.shape)
        for i in range(0, input_array.shape[0]):
            sum_matrix = np.zeros([input_array.shape[1], input_array.shape[2]])
            for j in range(max(0, i-self.n/2), min(self.input_array.shape[0]-1, i+self.n/2)):
                sum_matrix += input_array[j,:,:] ** 2
            self.divisor_matrix[i,:,:] = self.k + self.alpha * sum_matrix
        self.divisor_matrix **= self.beta
        self.output_array = self.input_array / self.divisor_matrix

    def get_output(self):
        return self.output_array

    def calc_layer_delta(self, downstream_layer):
        self.delta_array = downstream_layer.get_transformed_delta() * self.divisor_matrix

    def update_weight(self, rate):
        pass

    def get_transformed_delta(self):
        return self.delta_array