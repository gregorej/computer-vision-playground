from datasets.vehicles import concat, crowdai, object_detect
import matplotlib.pyplot as plt
import numpy as np
import cv2

ds = concat([object_detect, crowdai])

training_gen = ds.generator(10)
batch_img, batch_mask = next(training_gen)
### Plotting generator output
for i in range(10):
    im = np.array(batch_img[i], dtype=np.uint8)
    im_mask = np.array(batch_mask[i], dtype=np.uint8)
    plt.subplot(1, 3, 1)
    plt.imshow(im)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(im_mask[:, :, 0])
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.bitwise_and(im, im, mask=im_mask))
    plt.axis('off')
    plt.show()