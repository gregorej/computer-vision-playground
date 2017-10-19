import cv2
import numpy as np
import copy


class DataSample(object):

    def __init__(self, image, bb_boxes):
        self.image = image
        self.bb_boxes = bb_boxes

    def resize(self, new_size):
        img_size = np.shape(self.image)
        self.image = cv2.resize(self.image, new_size)
        img_size_post = np.shape(self.image)
        self.bb_boxes = map(lambda bbox: (
            bbox[0] * img_size_post[1] // img_size[1],
            bbox[1] * img_size_post[0] // img_size[0],
            bbox[2] * img_size_post[1] // img_size[1],
            bbox[3] * img_size_post[0] // img_size[0]
        ), self.bb_boxes)

    def merged_mask(self):
        img = self.image
        bb_boxes = self.bb_boxes
        img_mask = np.zeros_like(img[:, :, 0])
        for bbox in bb_boxes:
            img_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        img_mask = np.reshape(img_mask, (np.shape(img_mask)[0], np.shape(img_mask)[1], 1))
        return img_mask

    def translate(self, translation_range):
        # Translation augmentation
        image = self.image

        tr_x = translation_range*np.random.uniform()-translation_range/2
        tr_y = translation_range*np.random.uniform()-translation_range/2

        trans_m = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
        rows, cols, channels = image.shape
        self.bb_boxes = map(lambda s: (s[0] + tr_x, s[1] + tr_y, s[2] + tr_x, s[3] + tr_y), self.bb_boxes)

        self.image = cv2.warpAffine(image, trans_m, (cols, rows))

    def copy(self):
        return DataSample(self.image.copy(), copy.deepcopy(self.bb_boxes))
