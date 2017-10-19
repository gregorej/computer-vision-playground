import cv2
import numpy as np
from .DataSample import DataSample
import copy


def stretch(scale_range):
    return lambda ds: perform_stretch(ds, scale_range)


def perform_stretch(data_sample, scale_range=80):
    # Stretching augmentation
    img = data_sample.image.copy()
    bb_boxes_f = copy.deepcopy(data_sample.bb_boxes)
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

    def stretch_bbox(bbox):
        xmin = int((bbox[0] - p1[0])//(p2[0]-p1[0])*img.shape[1])
        ymin = int((bbox[1] - p1[1])//(p3[1]-p1[1])*img.shape[0])
        xmax = int((bbox[2] - p1[0])//(p2[0]-p1[0])*img.shape[1])
        ymax = int((bbox[3] - p1[1])//(p3[1]-p1[1])*img.shape[0])
        return xmin, ymin, xmax, ymax

    bb_boxes_f = map(stretch_bbox, bb_boxes_f)
    return DataSample(img, bb_boxes_f)


def flip_horizontally(ds):
    bb_boxes = copy.deepcopy(ds.bb_boxes)
    _, width, _ = np.shape(ds.image)
    middle = width // 2
    flip_x = lambda x: 2 * middle - x
    bb_boxes = map(lambda bbox: (
        flip_x(bbox[2]),
        bbox[1],
        flip_x(bbox[0]),
        bbox[3]
    ), bb_boxes)
    img = cv2.flip(ds.image, 1)

    return DataSample(img, bb_boxes)
