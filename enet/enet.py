from keras.layers.core import Activation
from keras.layers import Input
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Reshape

import decoder
import encoder


def autoencoder(nc, input_shape,
                loss='categorical_crossentropy',
                optimizer='adadelta'):
    data_shape = input_shape[0] * input_shape[1]
    inp = Input(shape=(input_shape[0], input_shape[1], 3))
    enet = encoder.build(inp)
    enet = decoder.build(enet, nc=nc, in_shape=input_shape)

    from keras import backend as K
    enet = Reshape((data_shape, nc))(enet)
    print K.int_shape(enet)
    enet = Activation('softmax')(enet)
    model = Model(inputs=inp, outputs=enet)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'mean_squared_error'])

    return model


if __name__ == "__main__":
    autoencoder = autoencoder(nc=2, input_shape=(512, 512))
    plot_model(autoencoder, to_file='{}.png'.format('enet'), show_shapes=True)
