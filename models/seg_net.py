from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Activation, Reshape, InputLayer, \
    ZeroPadding2D
from keras.layers.normalization import BatchNormalization


def seg_net(img_size, categories_count, filter_count=64):
    reduction_size = (2, 2)
    nb_encoders = 4

    model = Sequential()

    model.add(InputLayer(input_shape=(img_size[0], img_size[1], 3)))
    output_shapes = [model.output_shape]
    # encoders
    for i in range(nb_encoders):
        actual_filter_size = filter_count * (2 ** i)
        model.add(Convolution2D(actual_filter_size, 3, 3, border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(reduction_size))
        output_shapes.append(model.output_shape)
    expected_decoder_shapes = output_shapes[::-1]

    #decoders
    for i in range(nb_encoders):
        expected_output_shape = expected_decoder_shapes[i + 1]
        padding = (abs(expected_output_shape[1] - model.output_shape[1] * 2) / 2,
                   abs(expected_output_shape[2] - model.output_shape[2] * 2) / 2)
        model.add(UpSampling2D(reduction_size))
        if padding[0] != 0 or padding[1] != 0:
            model.add(ZeroPadding2D(padding=padding))
        actual_filter_size = filter_count * (2 ** (3 - i))
        model.add(Convolution2D(actual_filter_size, 3, 3, border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    model.add(Convolution2D(categories_count, 1, 1, border_mode='valid',))
    model.add(Reshape((img_size[0] * img_size[1], categories_count)))
    model.add(Activation('softmax'))

    return model

