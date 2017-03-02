import pandas as pd
import numpy as np
import cv2
from augmentation import trans_image, stretch_image, augment_brightness_camera_images, get_mask_seg
import os

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
            name_str, img, bb_boxes = get_image_by_name(i_line,
                                                        size=(img_cols, img_rows),
                                                        augmentation=True,
                                                        trans_range=50,
                                                        scale_range=50
                                                        )
            img_mask = get_mask_seg(img, bb_boxes)
            batch_images[i_batch] = img
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
            name_str, img, bb_boxes = get_image_by_name(i_line,
                                                        size=(img_cols, img_rows),
                                                        augmentation=False,
                                                        trans_range=0,
                                                        scale_range=0
                                                        )
            img_mask = get_mask_seg(img, bb_boxes)
            batch_images[i_batch] = img
            batch_masks[i_batch] = img_mask
        yield batch_images, batch_masks


def get_image_by_name(ind, size=(640, 300), augmentation=False, trans_range=20, scale_range=20):
    # Get image by name

    df = vehicles
    file_name = df['File_Path'][ind]
    img = cv2.imread(file_name)
    img_size = np.shape(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    name_str = file_name.split('/')
    name_str = name_str[-1]
    bb_boxes = df[df['Frame'] == name_str].reset_index()
    img_size_post = np.shape(img)

    if augmentation:
        img, bb_boxes = trans_image(img, bb_boxes, trans_range)
        img, bb_boxes = stretch_image(img, bb_boxes, scale_range)
        img = augment_brightness_camera_images(img)

    bb_boxes['xmin'] = np.round(bb_boxes['xmin'] / img_size[1] * img_size_post[1])
    bb_boxes['xmax'] = np.round(bb_boxes['xmax'] / img_size[1] * img_size_post[1])
    bb_boxes['ymin'] = np.round(bb_boxes['ymin'] / img_size[0] * img_size_post[0])
    bb_boxes['ymax'] = np.round(bb_boxes['ymax'] / img_size[0] * img_size_post[0])
    bb_boxes['Area'] = (bb_boxes['xmax'] - bb_boxes['xmin']) * (bb_boxes['ymax'] - bb_boxes['ymin'])

    return name_str, img, bb_boxes


if __name__ == '__main__':
    print vehicles.head()
    print vehicles.tail()
