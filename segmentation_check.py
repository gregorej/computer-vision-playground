from datasets import CamVid
from models import seg_net, enet
import cv2
import numpy as np
from image_util import normalize_histogram, resize_image


ds = CamVid.load_from_datasets_dir()
img_size = ds.img_size
labels = ds._label_dict


def bgr(rgb):
    return rgb[2], rgb[1], rgb[0]


def draw_label_captions(labels):
    square_size = 10
    row_height = 2 * square_size
    result_width = 200
    result = np.zeros((len(labels) * row_height, result_width, 4), dtype='uint8')
    row = 0
    row_margin = (row_height - square_size) / 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    for color, label in labels.iteritems():
        count, txt = label
        cv2.rectangle(result, (row_margin, row_height * row + row_margin), (square_size + row_margin, row_height * row + square_size + row_margin), bgr(color), thickness=cv2.FILLED)
        cv2.putText(result, txt, (square_size + 10, row_height * row + square_size + row_margin), font, 0.5, white, 1, cv2.LINE_AA)
        row += 1
    return result

cv2.imshow('legend', draw_label_captions(labels))
cv2.waitKey()


model = enet(ds.img_size, len(labels))

weights_path = 'trained_nets/enet.20170506-03:34:20.weights.h5'
model.load_weights(weights_path)

print('model loaded')

video_path = '/home/sharky/Wideo/krakow_walk.mp4'


reverse_labels = {}

for color, label in labels.iteritems():
    count, txt = label
    reverse_labels[count] = color


def colorize_labels(predicted):
    res = np.zeros((img_size[0], img_size[1], 3), dtype='uint8')
    for x in range(img_size[0]):
        for y in range(img_size[1]):
            predicted_label = np.argmax(predicted[x, y])
            rgb = reverse_labels[predicted_label]
            res[x, y] = bgr(rgb)
    return res



cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
count = 0
paused = False
while ret:
    if paused:
        key = cv2.waitKey()
        if key == ord('p'):
            paused = False
    elif count % 10 == 0:
        batch = np.zeros((1, img_size[0], img_size[1], 3))
        resized = normalize_histogram(resize_image(frame, img_size))
        batch[0] = resized
        predicted = model.predict(batch, batch_size=1)[0]
        predicted = np.reshape(predicted, (img_size[0], img_size[1], len(labels)))
        print(predicted.shape)
        cv2.imshow('original', resized)
        cv2.imshow('segmented', colorize_labels(predicted))
        key = cv2.waitKey(1)
        if key == ord('p'):
            paused = True
        print('frame ' + str(count) + ' analyzed')
    count += 1
    ret, frame = cap.read()
