from datasets import CamVid
from models import seg_net, enet
import cv2
import numpy as np
from image_util import normalize_histogram, resize_image


ds = CamVid.from_dir()
img_size = ds.img_size
labels = ds._label_dict


model = enet(ds.img_size, len(labels))

weights_path = 'trained_nets/enet.20170506-03:34:20.weights.h5'
model.load_weights(weights_path)

print 'model loaded'

video_path = '/home/sharky/Wideo/krakow_walk.mp4'


reverse_labels = {}

for color, label in labels.iteritems():
    count, txt = label
    reverse_labels[count] = color


def unlabel(predicted):
    res = np.zeros((img_size[0], img_size[1], 3), dtype='uint8')
    for x in range(img_size[0]):
        for y in range(img_size[1]):
            predicted_label = np.argmax(predicted[x, y])
            rgb = reverse_labels[predicted_label]
            res[x, y] = (rgb[2], rgb[1], rgb[0])
    return res

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
count = 0
while ret:
    if count % 10 == 0:
        batch = np.zeros((1, img_size[0], img_size[1], 3))
        resized = normalize_histogram(resize_image(frame, img_size))
        batch[0] = resized
        predicted = model.predict(batch, batch_size=1)[0]
        predicted = np.reshape(predicted, (img_size[0], img_size[1], len(labels)))
        print predicted.shape
        cv2.imshow('original', resized)
        cv2.imshow('segmented', unlabel(predicted))
        cv2.waitKey(1)
        print 'frame ' + str(count) + ' analyzed'
    count += 1
    ret, frame = cap.read()
