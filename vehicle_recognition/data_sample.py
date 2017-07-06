import cv2
import numpy as np


class DataSample(object):

    def __init__(self, image, bb_boxes):
        self.image = image
        self.bb_boxes = bb_boxes

    def resize(self, new_size):
        img_size = np.shape(self.image)
        self.image = cv2.resize(self.image, new_size)
        img_size_post = np.shape(self.image)
        self.bb_boxes['xmin'] = np.round(self.bb_boxes['xmin'] * img_size_post[1] / img_size[1])
        self.bb_boxes['xmax'] = np.round(self.bb_boxes['xmax'] * img_size_post[1] / img_size[1])
        self.bb_boxes['ymin'] = np.round(self.bb_boxes['ymin'] * img_size_post[0] / img_size[0])
        self.bb_boxes['ymax'] = np.round(self.bb_boxes['ymax'] * img_size_post[0] / img_size[0])
        self.bb_boxes['Area'] = (self.bb_boxes['xmax'] - self.bb_boxes['xmin']) * (self.bb_boxes['ymax'] - self.bb_boxes['ymin'])

    def merged_mask(self):
        img = self.image
        bb_boxes = self.bb_boxes
        img_mask = np.zeros_like(img[:, :, 0])
        for i in range(len(bb_boxes)):
            # plot_bbox(bb_boxes,i,'g')
            bb_box_i = [bb_boxes.iloc[i]['xmin'], bb_boxes.iloc[i]['ymin'],
                        bb_boxes.iloc[i]['xmax'], bb_boxes.iloc[i]['ymax']]
            img_mask[bb_box_i[1]:bb_box_i[3], bb_box_i[0]:bb_box_i[2]] = 1
            img_mask = np.reshape(img_mask, (np.shape(img_mask)[0], np.shape(img_mask)[1], 1))
        return img_mask

    def translate(self, translation_range):
        # Translation augmentation
        bb_boxes_f = self.bb_boxes
        image = self.image

        tr_x = translation_range*np.random.uniform()-translation_range/2
        tr_y = translation_range*np.random.uniform()-translation_range/2

        trans_m = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
        rows, cols, channels = image.shape
        bb_boxes_f['xmin'] += tr_x
        bb_boxes_f['xmax'] += tr_x
        bb_boxes_f['ymin'] += tr_y
        bb_boxes_f['ymax'] += tr_y

        self.image = cv2.warpAffine(image, trans_m, (cols, rows))

    def stretch(self, scale_range):
        # Stretching augmentation
        img = self.image
        bb_boxes_f = self.bb_boxes
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

        bb_boxes_f['xmin'] = (bb_boxes_f['xmin'] - p1[0])/(p2[0]-p1[0])*img.shape[1]
        bb_boxes_f['xmax'] = (bb_boxes_f['xmax'] - p1[0])/(p2[0]-p1[0])*img.shape[1]
        bb_boxes_f['ymin'] = (bb_boxes_f['ymin'] - p1[1])/(p3[1]-p1[1])*img.shape[0]
        bb_boxes_f['ymax'] = (bb_boxes_f['ymax'] - p1[1])/(p3[1]-p1[1])*img.shape[0]
        self.image = img

    def copy(self):
        return DataSample(self.image.copy(), self.bb_boxes.copy(deep=True))