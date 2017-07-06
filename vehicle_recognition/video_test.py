import vehicles_dataset as ds
import models
import sys
import time
import numpy as np
import cv2
from scipy.ndimage.measurements import label

model = models.small_unet(ds.img_size)
weights_path = 'model_weights.h5'
if len(sys.argv) > 2:
    weights_path = sys.argv[2]
model.load_weights(weights_path)


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        if (np.max(nonzeroy)-np.min(nonzeroy) > 50) & (np.max(nonzerox)-np.min(nonzerox) > 50):
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def draw_bounding_box(img, pred):
    # Take in RGB image
    img = np.array(img, dtype=np.uint8)
    img_pred = np.array(255*pred[0], dtype=np.uint8)
    heatmap = img_pred[:, :, 0]
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img


def segmentation(img, pred):
    im = np.array(img, dtype= np.uint8)
    im_pred = np.array(255*pred[0], dtype=np.uint8)
    rgb_mask_pred = cv2.cvtColor(im_pred, cv2.COLOR_GRAY2RGB)
    rgb_mask_pred[:, :, 1:3] = 0*rgb_mask_pred[:, :, 1:2]
    return cv2.addWeighted(rgb_mask_pred, 0.55, im, 1, 0)


def test_frame(frame):
    frame = cv2.resize(frame, (ds.img_cols, ds.img_rows))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.reshape(frame, (1, ds.img_rows, ds.img_cols, 3))
    pred = model.predict(frame)
    return pred, frame[0]

frame_count = 0

cap = cv2.VideoCapture(sys.argv[1])
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % 10 == 0:
        t = time.time()
        pred, img = test_frame(frame)
        end = time.time()
        print (end - t)
        cv2.imshow('car_detect', draw_bounding_box(segmentation(img, pred), pred))
    frame_count += 1
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # im = np.array(batch_img[i], dtype=np.uint8)

