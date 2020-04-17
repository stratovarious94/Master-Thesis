import math
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import get_file
from tensorflow.python.keras import initializers
from tensorflow.python.keras import backend as K
from keras_applications.imagenet_utils import preprocess_input as _preprocess


class EfficientNetConvInitializer(initializers.Initializer):
    """
    Custom initializer for the convolutions
    """
    def __init__(self):
        super(EfficientNetConvInitializer, self).__init__()

    def __call__(self, shape, dtype=None):
        dtype = dtype or K.floatx()
        kernel_height, kernel_width, _, out_filters = shape
        fan_out = int(kernel_height * kernel_width * out_filters)
        return tf.random.normal(shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


class EfficientNetDenseInitializer(initializers.Initializer):
    """
    Custom initializer for the dense layers
    """
    def __init__(self):
        super(EfficientNetDenseInitializer, self).__init__()

    def __call__(self, shape, dtype=None):
        dtype = dtype or K.floatx()
        init_range = 1.0 / np.sqrt(shape[1])
        return tf.random.uniform(shape, -init_range, init_range, dtype=dtype)


class Swish(layers.Layer):
    """
    Swish activation function. (Not implemented in tensorflow library yet)
    """
    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, training=None):
        return tf.nn.swish(inputs)


class DropConnect(layers.Layer):
    """
    DropConnect feature to get rid of unwanted parameters in the feature vectors
    """
    def __init__(self, drop_connect_rate=0., **kwargs):
        super(DropConnect, self).__init__(**kwargs)
        self.drop_connect_rate = float(drop_connect_rate)

    def call(self, inputs, training=None):
        def drop_connect():
            keep_prob = 1.0 - self.drop_connect_rate
            batch_size = tf.shape(inputs)[0]
            random_tensor = keep_prob
            random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            output = (inputs / keep_prob) * binary_tensor
            return output
        return K.in_train_phase(drop_connect, inputs, training=training)

    def get_config(self):
        config = {'drop_connect_rate': self.drop_connect_rate}
        base_config = super(DropConnect, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BlockArgs(object):
    """
    Instructions for building the EfficientNetB5
    """
    def __init__(self, input_filters=None, output_filters=None, kernel_size=None, strides=None, num_repeat=None,
                 se_ratio=None, expand_ratio=None, identity_skip=True):

        self.input_filters = input_filters
        self.output_filters = output_filters
        self.kernel_size=kernel_size
        self.strides = strides
        self.num_repeat = num_repeat
        self.se_ratio = se_ratio
        self.expand_ratio = expand_ratio
        self.identity_skip = identity_skip

    def get_default_block_list(self):
        DEFAULT_BLOCK_LIST = [
            BlockArgs(32, 16, kernel_size=3, strides=(1, 1), num_repeat=1, se_ratio=0.25, expand_ratio=1),
            BlockArgs(16, 24, kernel_size=3, strides=(2, 2), num_repeat=2, se_ratio=0.25, expand_ratio=6),
            BlockArgs(24, 40, kernel_size=5, strides=(2, 2), num_repeat=2, se_ratio=0.25, expand_ratio=6),
            BlockArgs(40, 80, kernel_size=3, strides=(2, 2), num_repeat=3, se_ratio=0.25, expand_ratio=6),
            BlockArgs(80, 112, kernel_size=5, strides=(1, 1), num_repeat=3, se_ratio=0.25, expand_ratio=6),
            BlockArgs(112, 192, kernel_size=5, strides=(2, 2), num_repeat=4, se_ratio=0.25, expand_ratio=6),
            BlockArgs(192, 320, kernel_size=3, strides=(1, 1), num_repeat=1, se_ratio=0.25, expand_ratio=6),
        ]

        return DEFAULT_BLOCK_LIST


class EfficientNetB5:
    """
    Main body of the model
    """
    def __init__(self, classes=1000, input=(224, 224, 3), weights='imagenet'):
        self.classes = classes
        self.input = input
        self.weights = weights

    def preprocess_input(self, x, data_format=None):
        return _preprocess(x, data_format, mode='torch', backend=K)

    def round_filters(self, filters, width_coefficient, depth_divisor, min_depth=None):
        multiplier = float(width_coefficient)
        divisor = int(depth_divisor)
        min_depth = min_depth
        if not multiplier:
            return filters

        filters *= multiplier
        min_depth = min_depth or divisor
        new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * filters:
            new_filters += divisor

        return int(new_filters)

    def round_repeats(self, repeats, depth_coefficient):
        multiplier = depth_coefficient
        if not multiplier:
            return repeats

        return int(math.ceil(multiplier * repeats))

    def SEBlock(self, input_filters, se_ratio, expand_ratio):
        num_reduced_filters = max(1, int(input_filters * se_ratio))
        filters = input_filters * expand_ratio
        spatial_dims = [1, 2]

        def block(inputs):
            x = inputs
            x = layers.Lambda(lambda a: K.mean(a, axis=spatial_dims, keepdims=True))(x)
            x = layers.Conv2D(num_reduced_filters, kernel_size=[1, 1], strides=[1, 1],
                              kernel_initializer=EfficientNetConvInitializer(), padding='same', use_bias=True)(x)
            x = Swish()(x)
            x = layers.Conv2D(filters, kernel_size=[1, 1], strides=[1, 1],
                              kernel_initializer=EfficientNetConvInitializer(), padding='same', use_bias=True)(x)
            x = layers.Activation('sigmoid')(x)
            out = layers.Multiply()([x, inputs])
            return out
        return block

    def MBConvBlock(self, input_filters, output_filters, kernel_size, strides, expand_ratio, se_ratio, id_skip,
                    drop_connect_rate, batch_norm_momentum=0.99, batch_norm_epsilon=1e-3, data_format=None):
        channel_axis = -1
        has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
        filters = input_filters * expand_ratio

        def block(inputs):
            if expand_ratio != 1:
                x = layers.Conv2D(filters, kernel_size=[1, 1], strides=[1, 1],
                                  kernel_initializer=EfficientNetConvInitializer(), padding='same',
                                  use_bias=False)(inputs)
                x = layers.BatchNormalization(axis=channel_axis, momentum=batch_norm_momentum,
                                              epsilon=batch_norm_epsilon)(x)
                x = Swish()(x)
            else:
                x = inputs

            x = layers.DepthwiseConv2D([kernel_size, kernel_size], strides=strides,
                                       depthwise_initializer=EfficientNetConvInitializer(), padding='same',
                                       use_bias=False)(x)
            x = layers.BatchNormalization(axis=channel_axis, momentum=batch_norm_momentum, epsilon=batch_norm_epsilon)(x)
            x = Swish()(x)
            if has_se:
                x = self.SEBlock(input_filters, se_ratio, expand_ratio)(x)
            x = layers.Conv2D(output_filters, kernel_size=[1, 1], strides=[1, 1],
                              kernel_initializer=EfficientNetConvInitializer(), padding='same', use_bias=False)(x)
            x = layers.BatchNormalization(axis=channel_axis, momentum=batch_norm_momentum,
                                          epsilon=batch_norm_epsilon)(x)
            if id_skip:
                if all(s == 1 for s in strides) and (input_filters == output_filters):
                    if drop_connect_rate:
                        x = DropConnect(drop_connect_rate)(x)
                    x = layers.Add()([x, inputs])
            return x
        return block

    def create(self, width_coefficient=1.6, depth_coefficient=2.2, dropout_rate=0.4, drop_connect_rate=0.,
               batch_norm_momentum=0.99, batch_norm_epsilon=1e-3, depth_divisor=8):

        data_format = 'channels_last'
        channel_axis = -1
        block_args_list = BlockArgs().get_default_block_list()
        stride_count = 1
        for block_args in block_args_list:
            if block_args.strides is not None and block_args.strides[0] > 1:
                stride_count += 1

        min_size = int(2 ** stride_count)

        inputs = layers.Input(shape=self.input)

        x = inputs
        x = layers.Conv2D(
            filters=self.round_filters(32, width_coefficient, depth_divisor), kernel_size=[3, 3],
            strides=[2, 2], kernel_initializer=EfficientNetConvInitializer(), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization(axis=channel_axis, momentum=batch_norm_momentum, epsilon=batch_norm_epsilon)(x)
        x = Swish()(x)

        num_blocks = sum([block_args.num_repeat for block_args in block_args_list])
        drop_connect_rate_per_block = drop_connect_rate / float(num_blocks)

        for block_idx, block_args in enumerate(block_args_list):
            assert block_args.num_repeat > 0
            block_args.input_filters = self.round_filters(block_args.input_filters, width_coefficient, depth_divisor)
            block_args.output_filters = self.round_filters(block_args.output_filters, width_coefficient, depth_divisor)
            block_args.num_repeat = self.round_repeats(block_args.num_repeat, depth_coefficient)
            x = self.MBConvBlock(block_args.input_filters, block_args.output_filters,
                                 block_args.kernel_size, block_args.strides,
                                 block_args.expand_ratio, block_args.se_ratio,
                                 block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                                 batch_norm_momentum, batch_norm_epsilon, data_format)(x)

            if block_args.num_repeat > 1:
                block_args.input_filters = block_args.output_filters
                block_args.strides = [1, 1]

            for _ in range(block_args.num_repeat - 1):
                x = self.MBConvBlock(block_args.input_filters, block_args.output_filters,
                                     block_args.kernel_size, block_args.strides,
                                     block_args.expand_ratio, block_args.se_ratio,
                                     block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                                     batch_norm_momentum, batch_norm_epsilon, data_format)(x)

        x = layers.Conv2D(filters=self.round_filters(1280, width_coefficient, depth_coefficient),
                          kernel_size=[1, 1], strides=[1, 1], kernel_initializer=EfficientNetConvInitializer(),
                          padding='same', use_bias=False)(x)
        x = layers.BatchNormalization(axis=channel_axis, momentum=batch_norm_momentum, epsilon=batch_norm_epsilon)(x)
        x = Swish()(x)
        x = layers.GlobalAveragePooling2D(data_format=data_format)(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

        outputs = x
        model = Model(inputs, outputs)

        if self.weights == 'imagenet':
            weights_path = get_file(
                'efficientnet-b5_notop.h5',
                "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b5_notop.h5",
                cache_subdir='models')
        model.load_weights(weights_path)

        x = model.output
        x = layers.Dense(self.classes, kernel_initializer=EfficientNetDenseInitializer())(x)
        x = layers.Activation('softmax')(x)

        outputs = x
        model = Model(inputs, outputs)
        # for layer in model.layers[:-150]:
        #     layer.trainable = False
        model.summary()
        return model
