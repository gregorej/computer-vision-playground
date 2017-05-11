import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from image_util import normalize_histogram, resize_image


def read_labels(labels_file):
    result = {}
    count = 0
    with open(labels_file, 'r') as f:
        for line in f:
            parts = line.split()
            result[(int(parts[0]), int(parts[1]), int(parts[2]))] = (count, parts[3])
            count += 1
    return result


class CamVidDataset(object):
    img_size = (720 / 2, 960 / 2)

    class Generator(object):

        def __init__(self, dataset, batch_size):
            self.__current_index = 0
            self.ds = dataset
            self.batch_size = batch_size

        def next(self):
            img_size = self.ds.img_size
            if self.__current_index >= self.ds.size:
                self.__current_index = 0
            actual_batch_size = min(self.batch_size, self.ds.size - self.__current_index)
            image_data = np.zeros((actual_batch_size, img_size[0], img_size[1], 3), dtype='uint8')
            categories_count = len(self.ds._label_dict)
            label_data = np.zeros((actual_batch_size, img_size[0] * img_size[1], categories_count), dtype='uint8')
            for idx in range(actual_batch_size):
                image, labels = self.ds[self.__current_index + idx]
                label_data[idx] = labels
                image_data[idx] = image
            self.__current_index += actual_batch_size
            return image_data, label_data

        def __iter__(self):
            return self

    class Iterator(object):

        def __init__(self, owner):
            self.__current_index = 0
            self.ds = owner

        def next(self):
            if self.__current_index >= self.ds.size:
                raise StopIteration
            result = self.ds[self.__current_index]
            self.__current_index += 1
            return result

    def _label_image_to_labels(self, label_image):
        categories_count = len(self._label_dict)
        res = np.zeros((self.img_size[0], self.img_size[1]), dtype='uint8')
        for color, label in self._label_dict.iteritems():
            bgr_color = [color[2], color[1], color[0]]
            res[np.where((label_image == bgr_color).all(axis=2))] = label[0]
        res = np.reshape(res, (self.img_size[0] * self.img_size[1]))

        return to_categorical(res, categories_count)

    def __init__(self, sample_files, dataset_home, label_dict):
        self._images_dir = dataset_home + '/701_StillsRaw_full'
        self._sample_files = sample_files
        self._labels_dir = dataset_home + '/LabeledApproved_full'
        self._dataset_home = dataset_home
        self._label_dict = label_dict
        self.size = len(self._sample_files)
        self.categories_count = len(self._label_dict)

    @classmethod
    def load_from_datasets_dir(cls):
        datasets_home = os.environ['DATASETS'] + '/camvid'
        images_dir = datasets_home + '/701_StillsRaw_full'
        labels_dir = datasets_home + '/LabeledApproved_full'
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        image_files.sort()
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.png')]
        label_files.sort()
        samples = zip(image_files, label_files)
        label_dict = read_labels(datasets_home + '/label_colors.txt')
        return CamVidDataset(samples, datasets_home, label_dict)

    def __len__(self):
        return self.size

    def __iter__(self):
        return self.Iterator(self)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return CamVidDataset(self._sample_files[item], self._dataset_home, self._label_dict)
        elif isinstance(item, int) and 0 <= item < self.size:
            image_file, label_file = self._sample_files[item]
            full_image_path = os.path.join(self._images_dir, image_file)
            image = normalize_histogram(resize_image(cv2.imread(full_image_path), self.img_size))
            full_label_path = os.path.join(self._labels_dir, label_file)
            labels = self._label_image_to_labels(resize_image(cv2.imread(full_label_path), self.img_size))
            return image, labels
        else:
            raise IndexError('invalid index: ' + str(item))

    def generator(self, batch_size=10):
        return self.Generator(self, batch_size)