from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Activation, Reshape, InputLayer, \
    ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from CamVidDataset import CamVidDataset


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


if __name__ == "__main__":
    camvid = CamVidDataset.from_dir()
    model = seg_net(camvid.img_size, camvid.categories_count)
    left_for_test = 1
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    test_set = camvid[0:-100]
    model.fit_generator(test_set.generator(1),
                        nb_epoch=2,
                        samples_per_epoch=len(test_set))

    evaluate_set = camvid[len(camvid) - 100:len(camvid)]
    print model.evaluate_generator(evaluate_set.generator(1), 100)

    model.save("seg_net.mine.h5")
    model.save_weights("seg_net.mine.weights.h5", overwrite=True)

