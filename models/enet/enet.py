from keras.layers.core import Activation
from keras.layers import Input
from keras.models import Model
from keras.layers import Reshape

import decoder
import encoder


def build(input_shape, categories_count):
    data_shape = input_shape[0] * input_shape[1]
    inp = Input(shape=(input_shape[0], input_shape[1], 3))
    enet = encoder.build(inp)
    enet = decoder.build(enet, nc=categories_count, in_shape=input_shape)

    from keras import backend as K
    enet = Reshape((data_shape, categories_count))(enet)
    print K.int_shape(enet)
    enet = Activation('softmax')(enet)
    model = Model(inputs=inp, outputs=enet)
    return model

