import pandas as pd
import numpy as np
import cv2
from augmentation import augment_brightness_camera_images
import os
from data_sample import DataSample

datasets_home = os.environ['DATASETS'] + '/vehicles'
# Image size,
img_rows = 640
img_cols = 960

img_size = (img_rows, img_cols)

dir_label = [datasets_home + '/object-dataset',
             datasets_home + '/object-detection-crowdai']

df_files1 = pd.read_csv(dir_label[1] + '/labels.csv', header=0)
df_vehicles1 = df_files1[(df_files1['Label'] == 'Car') | (df_files1['Label'] == 'Truck')].reset_index()
df_vehicles1 = df_vehicles1.drop('index', 1)
df_vehicles1['File_Path'] = dir_label[1] + '/' + df_vehicles1['Frame']
df_vehicles1 = df_vehicles1.drop('Preview URL', 1)
df_vehicles1.head()

df_files2 = pd.read_csv(datasets_home + '/object-dataset/labels.csv', header=0, sep=' ')
df_vehicles2 = df_files2[(df_files2['Label'] == 'car') | (df_files2['Label'] == 'truck')].reset_index()
df_vehicles2 = df_vehicles2.drop('index', 1)
df_vehicles2 = df_vehicles2.drop('RM', 1)
df_vehicles2 = df_vehicles2.drop('ind', 1)

df_vehicles2['File_Path'] = dir_label[0] + '/' + df_vehicles2['Frame']

df_vehicles2.head()

# combine data frames
vehicles = pd.concat([df_vehicles1, df_vehicles2]).reset_index()
vehicles = vehicles.drop('index', 1)
vehicles.columns = ['File_Path', 'Frame', 'Label', 'ymin', 'xmin', 'ymax', 'xmax']


# Training generator, generates augmented images
def generate_train_batch(batch_size=32):
    data = vehicles
    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_masks = np.zeros((batch_size, img_rows, img_cols, 1))
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data) - 2000)
            sample = get_sample_by_index(i_line)
            sample.resize((img_cols, img_rows))
            # sample.translate(50)
            # sample.stretch(50)
            sample.image = augment_brightness_camera_images(sample.image)
            img_mask = sample.merged_mask()
            batch_images[i_batch] = sample.image
            batch_masks[i_batch] = img_mask
        yield batch_images, batch_masks


# Testing generator, generates augmented images
def generate_test_batch(batch_size=32):
    data = vehicles
    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_masks = np.zeros((batch_size, img_rows, img_cols, 1))
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(2000)
            i_line = i_line + len(data) - 2000
            sample = get_sample_by_index(i_line)
            sample.resize((img_cols, img_rows))
            img_mask = sample.merged_mask()
            batch_images[i_batch] = sample.image
            batch_masks[i_batch] = img_mask
        yield batch_images, batch_masks


def get_sample_by_index(index):
    # Get image by name

    df = vehicles
    file_name = df['File_Path'][index]
    img = cv2.imread(file_name)
    name_str = file_name.split('/')
    name_str = name_str[-1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bb_boxes = df[df['Frame'] == name_str].reset_index()
    return DataSample(img, bb_boxes)

if __name__ == '__main__':
    print(vehicles.head())
    print(vehicles.tail())
