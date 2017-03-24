from keras.models import Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Activation, Reshape, InputLayer, \
    ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from CamVidDataset import CamVidDataset


def seg_net(img_size, categories_count):
    reduction_size = (2, 2)

    def add_encoding_part(model):
        model.add(Convolution2D(1, 3, 3, border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(reduction_size))
        print model.output_shape

    def add_decoding_part(model, index=0):
        model.add(UpSampling2D(reduction_size))
        if index == 2:
            model.add(ZeroPadding2D(padding=(1, 0)))
        model.add(Convolution2D(1, 3, 3, border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        print model.output_shape

    model = Sequential()

    model.add(InputLayer(input_shape=(img_size[0], img_size[1], 3)))
    add_encoding_part(model)
    add_encoding_part(model)
    add_encoding_part(model)
    add_encoding_part(model)
    add_encoding_part(model)

    # upsampling phase
    add_decoding_part(model, 0)
    add_decoding_part(model, 1)
    add_decoding_part(model, 2)
    add_decoding_part(model, 3)
    add_decoding_part(model, 4)

    model.add(Convolution2D(categories_count, 1, 1, border_mode='valid',))
    model.add(Reshape((img_size[0] * img_size[1], categories_count)))
    print model.output_shape
    model.add(Activation('softmax'))
    print model.output_shape

    return model

camvid = CamVidDataset.from_dir()
model = seg_net(camvid.img_size, camvid.categories_count)
left_for_test = 1
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit_generator(camvid.generator(),
                    nb_epoch=1,
                    samples_per_epoch=len(camvid))

model.save('seg_net.mine.h5')
model.save_weights("seg_net.mine.weights.h5", overwrite=True)

print model.evaluate_generator(camvid[len(camvid) - 100:len(camvid)].generator(1), 100)
