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

        def __init__(self, owner, batch_size):
            self.__current_index = 0
            self.owner = owner
            self.batch_size = batch_size

        def next(self):
            img_size = self.owner.img_size
            if self.__current_index >= self.owner.size:
                self.__current_index = 0
            actual_batch_size = min(self.batch_size, self.owner.size - self.__current_index)
            image_data = np.zeros((actual_batch_size, img_size[0], img_size[1], 3), dtype='uint8')
            label_data = np.zeros((actual_batch_size, img_size[0], img_size[1]), dtype='uint8')
            for idx in range(0, actual_batch_size):
                image, labels = self.owner[self.__current_index + idx]
                label_data[idx] = labels
                image_data[idx] = image
            self.__current_index += actual_batch_size
            return image_data, label_data

        def __iter__(self):
            return self

    class Iterator(object):

        def __init__(self, owner):
            self.__current_index = 0
            self.owner = owner

        def next(self):
            if self.__current_index >= self.owner.size:
                raise StopIteration
            result = self.owner[self.__current_index]
            self.__current_index += 1
            return result

    def __init__(self, image_files, label_files, dataset_home):
        self._images_dir = dataset_home + '/701_StillsRaw_full'
        self._images = image_files
        self._images.sort()
        self._labels_dir = dataset_home + '/LabeledApproved_full'
        self._labels = label_files
        self._labels.sort()
        self._dataset_home = dataset_home
        self.size = len(self._images)

    @classmethod
    def from_dir(cls):
        datasets_home = os.environ['DATASETS'] + '/camvid'
        images_dir = datasets_home + '/701_StillsRaw_full'
        labels_dir = datasets_home + '/LabeledApproved_full'
        return CamVidDataset([f for f in os.listdir(images_dir) if f.endswith('.png')],
                             [f for f in os.listdir(labels_dir) if f.endswith('.png')], datasets_home)

    def __len__(self):
        return self.size

    def __iter__(self):
        return self.Iterator(self)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return CamVidDataset(self._images[item], self._labels[item], self._dataset_home)
        if isinstance(item, int) and 0 <= item < self.size:
            full_image_path = os.path.join(self._images_dir, self._images[item])
            image = cv2.imread(full_image_path)
            full_label_path = os.path.join(self._labels_dir, self._labels[item])
            labels = cv2.cvtColor(cv2.imread(full_label_path), cv2.COLOR_BGR2GRAY)
            return image, labels
        else:
            raise IndexError('invalid index')

    def generator(self, batch_size=10):
        return self.Generator(self, batch_size)

camvid = CamVidDataset.from_dir()[0:-100]
model = seg_net(camvid.img_size)
left_for_test = 1
model.compile(loss='mean_squared_error', optimizer='sgd')
model.fit_generator(camvid.generator(),
                    nb_epoch=1,
                    samples_per_epoch=len(camvid))

model.save('seg_net.mine.h5')
model.save_weights("seg_net.mine.weights.h5", overwrite=True)

print model.evaluate_generator(camvid[len(camvid) - 100:len(camvid)].generator(10), 10)