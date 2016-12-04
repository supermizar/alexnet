from conv_layer import ConvLayer


class ConvLayerHidden(ConvLayer):
    def __init__(self, network, filter_width,
                 filter_height, filter_number,
                 zero_padding, stride, activator,
                 learning_rate):
        upstream_layer = network.layers[-1]
        input_width = upstream_layer.output_width
        input_height = upstream_layer.output_height
        channel_number = upstream_layer.filter_number
        ConvLayer.__init__(self, network, input_width, input_height,
                           channel_number, filter_width,
                           filter_height, filter_number,
                           zero_padding, stride, activator,
                           learning_rate)
