from keras.models import Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Activation, Reshape, InputLayer, \
    ZeroPadding2D
from keras.layers.normalization import BatchNormalization
import os
import cv2
import numpy as np


def seg_net(img_size):
    reduction_size = (2, 2)

    def add_encoding_part(model):
        model.add(Convolution2D(1, 3, 3, border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(reduction_size))
        print model.output_shape

    def add_decoding_part(model, index=0):
        model.add(UpSampling2D(reduction_size))
        if index == 1:
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

    model.add(Reshape((img_size[0], img_size[1])))
    print model.output_shape
    model.add(Activation('softmax'))

    return model


class CamVidDataset(object):
    img_size = (720, 960)

    class Generator(object):

        def __init__(self, owner):
            self.__current_index = 0
            self.owner = owner

        def next(self):
            img_size = self.owner.img_size
            if self.__current_index >= self.owner.size:
                self.__current_index = 0
            actual_batch_size = min(self.owner.batch_size, self.owner.size - self.__current_index)
            image_data = np.zeros((actual_batch_size, img_size[0], img_size[1], 3), dtype='uint8')
            label_data = np.zeros((actual_batch_size, img_size[0], img_size[1]), dtype='uint8')
            for idx, img_file in enumerate(self.owner._images[self.__current_index:actual_batch_size]):
                full_path = os.path.join(self.owner._images_dir, img_file)
                image_data[idx] = cv2.imread(full_path)
            for idx, img_file in enumerate(self.owner._labels[self.__current_index:actual_batch_size]):
                full_path = os.path.join(self.owner._labels_dir, img_file)
                label_data[idx] = cv2.cvtColor(cv2.imread(full_path), cv2.COLOR_BGR2GRAY)
            self.__current_index += actual_batch_size
            return image_data, label_data

        def __iter__(self):
            return self

    def __init__(self, batch_size):
        self.__current_index = 0
        self.batch_size = batch_size
        datasets_home = os.environ['DATASETS'] + '/camvid'
        self._images_dir = datasets_home + '/701_StillsRaw_full'
        self._images = [f for f in os.listdir(self._images_dir) if f.endswith('.png')]
        self._images.sort()
        self._labels_dir = datasets_home + '/LabeledApproved_full'
        self._labels = [f for f in os.listdir(self._labels_dir) if f.endswith('.png')]
        self._labels.sort()
        self.size = len(self._images)

    def __len__(self):
        return self.size

    def generator(self):
        return self.Generator(self)


camvid = CamVidDataset(10)
model = seg_net(camvid.img_size)
left_for_test = 1
model.compile(loss='mean_squared_error', optimizer='sgd')
model.fit_generator(camvid.generator(),
                    nb_epoch=1,
                    samples_per_epoch=len(camvid))

model.save('seg_net.mine.h5')
model.save_weights("seg_net.mine.weights.h5", overwrite=True)