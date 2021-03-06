from conv_layer import ConvLayer


class ConvLayerHidden(ConvLayer):
    def __init__(self, network, filter_width,
                 filter_height, filter_number,
                 zero_padding, stride, activator,
                 learning_rate, momentum_rate=0.0, decay_rate=0.0):
        upstream_layer = network.layers[-1]
        input_width = upstream_layer.get_output().shape[1]
        input_height = upstream_layer.get_output().shape[2]
        channel_number = upstream_layer.get_output().shape[0]
        ConvLayer.__init__(self, network, input_width, input_height,
                           channel_number, filter_width,
                           filter_height, filter_number,
                           zero_padding, stride, activator,
                           learning_rate, momentum_rate=momentum_rate, decay_rate=decay_rate)


