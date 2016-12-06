from math import exp


class ReluActivator(object):
    def forward(self, weighted_input):
        # return weighted_input
        return max(0, weighted_input)

    def backward(self, output):
        return 1 if output > 0 else 0


class Softmax(object):
    def forward(self, weighted_input_array):
        sum_e_net = reduce(lambda ret, net: ret + exp(net), weighted_input_array, 0)
        return [exp(x) / sum_e_net for x in weighted_input_array]
