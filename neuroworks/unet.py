# implementing the U-Net following the first publication:
# TODO URL
# implementation inspired by:
# TODO Github of tf-unet

import tensorflow as tf
import numpy as np
from collections import OrderedDict

from model import Model
import layers

class Unet(Model):

    def architecture(self):
        """
        The U-Net architecture.
        @returns: Logit output of last layer and all variables.
        """

        # layer parameters
        n_layers = self.model_params.get('n_layers', 3)
        initial_feature_size = self.model_params.get('initial_feature_size', 16)
        filter_size = self.model_params.get('filter_size',3)
        pool_size = self.model_params.get('pool_size',2)
        channels = self.n_channels

        nx = tf.shape(self.x)[1]
        ny = tf.shape(self.x)[2]

        x_image = tf.reshape(self.x, tf.pack([-1,nx,ny,self.n_channels]))
        in_node = x_image
        batch_size = tf.shape(x_image)[0]

        weights = []
        biases  = []
        convs   = []

        pools  = OrderedDict()
        deconv = OrderedDict()
        down_outputs = OrderedDict()
        up_outputs = OrderedDict()

        # down layers
        for layer in xrange(n_layers):
            features = 2**layer*initial_feature_size
            stddev = np.sqrt(2 / (filter_size**2 * features))

            # weights for first convolution
            if layer == 0:
                w1 = layers.weight_variable([filter_size, filter_size, channels, features],
                        stddev)
            else:
                w1 = layers.weight_variable([filter_size, filter_size, features//2, features],
                        stddev)

            w2 = layers.weight_variable([filter_size, filter_size, features, features],
                    stddev) # weights for second convolution
            b1 = layers.bias_variable([features])
            b2 = layers.bias_variable([features])

            # TODO where exactly should we use dropout ?
            conv1 = layers.conv2d_dropout(in_node, w1, self.keep_prob)
            conv2 = layers.conv2d_dropout(tf.nn.relu(conv1 + b1), w2, self.keep_prob)

            # input for downsampling layers and skipconnections
            down_outputs[layer] = tf.nn.relu(conv2 + b2)

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

            # down-sampling with max-pooling
            if layer < n_layers-1:
                pools[layer] = layers.max_pool2d(down_outputs[layer], pool_size)
                in_node = pools[layer]

        in_node = down_outputs[n_layers-1]

        # up layers, range from [layers-2, 0]
        for layer in range(n_layers-2, -1, -1):
            features = 2**(layer+1)*initial_feature_size
            stddev = np.sqrt(2 / (filter_size**2 * features))

            # weight for upsampling with deconvolution
            wd = layers.weight_variable([pool_size, pool_size, features//2, features],
                    stddev)
            # bias variables for upsampling
            bd = layers.bias_variable([features//2])

            # upsampling with deconvolution
            h_deconv = tf.nn.relu(layers.deconv2d(in_node, wd, pool_size) + bd)

            # concatenate upsampling with skiplayer
            h_deconv_concat = layers.crop_and_concat(down_outputs[layer], h_deconv)
            deconv[layer] = h_deconv_concat

            # weights for the in-layer convolutions
            w1 = layers.weight_variable([filter_size, filter_size, features, features//2],
                    stddev)
            w2 = layers.weight_variable([filter_size, filter_size, features//2, features//2],
                    stddev)

            b1 = layers.bias_variable([features//2])
            b2 = layers.bias_variable([features//2])

            # TODO if I understand the U-Net paper right, we shouldn't use dropout here, check with Nasim
            # in-layer convolutions
            conv1 = layers.conv2d(h_deconv_concat, w1)
            conv2 = layers.conv2d(tf.nn.relu(conv1 + b1), w2)
            in_node = tf.nn.relu(conv2 + b2)
            up_outputs[layer] = in_node

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

        # Output Map
        weight = layers.weight_variable([1, 1, initial_feature_size, 2], stddev)
        bias   = layers.bias_variable([2])
        conv   = layers.conv2d(in_node, weight)
        output_map = tf.nn.relu(conv + bias)
        up_outputs["out"] = output_map

        variables = []
        for w1,w2 in weights:
            variables.append(w1)
            variables.append(w2)

        for b1,b2 in biases:
            variables.append(b1)
            variables.append(b2)

        return output_map, variables



    def model_params_descriptions(self):
        """
        Returns a dictionary containing the expected model param names and a tuple with their description and default value.
        """
        return {'n_layers' : ('Number of layers. The total number of layers is 2*(n_layers) + 1 this number (UP + downscaling)',3),
                'initial_feature_size' : ('Number of features in the first layer.',16),
                'filter_size' : ('Size of convolutional filter',3),
                'pool_size'   : ('Pooling factor',2)}
