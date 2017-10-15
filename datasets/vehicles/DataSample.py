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
            bbox[0] * img_size_post[1] / img_size[1],
            bbox[1] * img_size_post[0] / img_size[0],
            bbox[2] * img_size_post[1] / img_size[1],
            bbox[3] * img_size_post[0] / img_size[0]
        ), self.bb_boxes)

    def merged_mask(self):
        img = self.image
        bb_boxes = self.bb_boxes
        img_mask = np.zeros_like(img[:, :, 0])
        for i in range(len(bb_boxes)):
            bb_box_i = bb_boxes[i]
            img_mask[bb_box_i[1]:bb_box_i[3], bb_box_i[0]:bb_box_i[2]] = 1
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

    def stretch(self, scale_range):
        # Stretching augmentation
        img = self.image
        # bb_boxes_f = bb_boxes_f.copy(deep=True)

        tr_x1 = scale_range*np.random.uniform()
        tr_y1 = scale_range*np.random.uniform()
        p1 = (tr_x1,tr_y1)
        tr_x2 = scale_range*np.random.uniform()
        tr_y2 = scale_range*np.random.uniform()
        p2 = (img.shape[1]-tr_x2,tr_y1)

        p3 = (img.shape[1]-tr_x2,img.shape[0]-tr_y2)
        p4 = (tr_x1,img.shape[0]-tr_y2)

        pts1 = np.float32([[p1[0],p1[1]],
                           [p2[0],p2[1]],
                           [p3[0],p3[1]],
                           [p4[0],p4[1]]])
        pts2 = np.float32([[0,0],
                           [img.shape[1],0],
                           [img.shape[1],img.shape[0]],
                           [0,img.shape[0]] ]
                          )

        m = cv2.getPerspectiveTransform(pts1,pts2)
        img = cv2.warpPerspective(img, m, (img.shape[1], img.shape[0]))
        img = np.array(img, dtype=np.uint8)

        self.bb_boxes = map(lambda bbox: (
            (bbox[0] - p1[0])/(p2[0]-p1[0])*img.shape[1], # xmin
            (bbox[2] - p1[0])/(p2[0]-p1[0])*img.shape[1], # xmax
            (bbox[1] - p1[1])/(p3[1]-p1[1])*img.shape[0], # ymin
            (bbox[3] - p1[1])/(p3[1]-p1[1])*img.shape[0], # ymax
        ), self.bb_boxes)
        self.image = img

    def copy(self):
        return DataSample(self.image.copy(), copy.deepcopy(self.bb_boxes))
