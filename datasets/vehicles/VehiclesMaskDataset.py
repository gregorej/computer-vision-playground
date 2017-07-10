import numpy as np
import os
import cv2
import pandas as pd
from DataSample import DataSample

datasets_home = os.environ['DATASETS'] + '/vehicles'


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

    def __init__(self, vehicle_rows):
        self.__vehicle_rows = vehicle_rows
        self.size = len(vehicle_rows.index)

    def __len__(self):
        return self.size

    def _get_sample_by_index(self, index):
        # Get image by name
        df = self.__vehicle_rows
        file_name = df['File_Path'][index]
        img = cv2.imread(file_name)
        name_str = file_name.split('/')
        name_str = name_str[-1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bb_boxes = df[df['Frame'] == name_str].reset_index()
        return DataSample(img, bb_boxes)

    def generator(self, batch_size=10):
        return self.Generator(self, batch_size)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return VehiclesMaskDataset(self.__vehicle_rows[item])
        else:
            raise IndexError('invalid index: ' + str(item))

    @staticmethod
    def concat(datasets):
        concat_vehicles = pd.concat(map(lambda ds: ds.__vehicle_rows, datasets)).reset_index()
        concat_vehicles = concat_vehicles.drop('index', 1)
        return VehiclesMaskDataset(concat_vehicles)

    @staticmethod
    def load_from_dir(directory, separator=' '):
        datasets_home = os.environ['DATASETS']
        absolute_directory = datasets_home + '/' + directory
        labels_csv = pd.read_csv(absolute_directory + '/labels.csv', header=0, delimiter=separator)
        filtered_labels_csv = labels_csv[(labels_csv['Label'].str.lower() == 'car') | (labels_csv['Label'].str.lower() == 'truck')].reset_index()
        filtered_labels_csv = filtered_labels_csv[filtered_labels_csv["xmin"] < filtered_labels_csv["xmax"]].reset_index()
        filtered_labels_csv = filtered_labels_csv.drop('index', 1)
        filtered_labels_csv['File_Path'] = absolute_directory + '/' + filtered_labels_csv['Frame']
        if 'Preview URL' in filtered_labels_csv.columns:
            filtered_labels_csv = filtered_labels_csv.drop('Preview URL', 1)
        return VehiclesMaskDataset(filtered_labels_csv)
