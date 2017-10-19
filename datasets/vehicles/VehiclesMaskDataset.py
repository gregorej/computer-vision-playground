import numpy as np
import cv2
from .DataSample import DataSample
from datasets.util import datasets_home, get_file_through_cache, load_bbox_samples, download_if_needed
import random
import copy


class VehiclesMaskDataset(object):

    img_rows = 640
    img_cols = 960

    img_size = (img_rows, img_cols)

    class Generator(object):

        def __init__(self, dataset, batch_size):
            self.__current_index = 0
            self.ds = dataset
            self.batch_size = batch_size

        def next(self):
            img_size = self.ds.img_size
            if self.__current_index >= self.ds.size:
                self.__current_index = 0
            actual_batch_size = min(self.batch_size, len(self.ds) - self.__current_index)
            image_data = np.zeros((actual_batch_size, img_size[0], img_size[1], 3), dtype='uint8')
            mask_data = np.zeros((actual_batch_size, img_size[0], img_size[1], 1), dtype='uint8')
            for idx in range(actual_batch_size):
                sample = self.ds._get_sample_by_index(idx + self.__current_index)
                sample.resize((self.ds.img_cols, self.ds.img_rows))
                mask_data[idx] = sample.merged_mask()
                image_data[idx] = sample.image
            self.__current_index += actual_batch_size
            return image_data, mask_data

        def __iter__(self):
            return self

    def __init__(self, vehicle_rows, augmentations=[]):
        self.__vehicle_rows = vehicle_rows
        self.size = len(vehicle_rows)
        self.augmentations = augmentations

    def with_augmentations(self, augmentations):
        return VehiclesMaskDataset(self.__vehicle_rows, augmentations)

    def __len__(self):
        return self.size * (len(self.augmentations) + 1)

    def _get_sample_by_index(self, index):
        # Get image by name
        original_image_index = index // (len(self.augmentations) + 1)
        augmentation_index = index % (len(self.augmentations) + 1)
        sample = self.__vehicle_rows[original_image_index]
        image_file_name = get_file_through_cache(sample[0])
        img = cv2.imread(image_file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bb_boxes = sample[1]
        sample = DataSample(img, bb_boxes)
        return sample if augmentation_index == 0 else self.augmentations[augmentation_index - 1](sample)

    def generator(self, batch_size=10):
        return self.Generator(self, batch_size)

    def shuffled(self):
        shuffled_samples = copy.deepcopy(self.__vehicle_rows)
        random.shuffle(shuffled_samples)
        return VehiclesMaskDataset(shuffled_samples)

    def merge(self, other_set):
        return VehiclesMaskDataset(self.__vehicle_rows + other_set.__vehicle_rows)

    def __getitem__(self, item):
        if isinstance(item, slice):
            augmentation_scale = (1 + len(self.augmentations))
            samples_slice = self.__vehicle_rows[item.start / augmentation_scale:item.stop / augmentation_scale]
            return VehiclesMaskDataset(samples_slice, self.augmentations)
        else:
            raise IndexError('invalid index: ' + str(item))

    @staticmethod
    def concat(datasets):
        result = VehiclesMaskDataset([])
        for dataset in datasets:
            result = result.merge(dataset)
        return result

    @staticmethod
    def load_from_dir(directory, separator=' '):
        absolute_directory = datasets_home + '/' + directory
        absolute_directory = download_if_needed(absolute_directory)
        samples = load_bbox_samples(absolute_directory, separator=separator)
        return VehiclesMaskDataset(samples)
