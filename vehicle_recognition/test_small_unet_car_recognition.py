import vehicles_dataset as ds
import models
import sys
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt


testing_gen = ds.generate_test_batch(20)

model = models.small_unet(ds.img_size)
weights_path = 'model_weights.h5'
if len(sys.argv) > 1:
    weights_path = sys.argv[1]
model.load_weights(weights_path)

# setting batch size too high might cause bad_alloc caused by running out of memory
training_gen = ds.generate_train_batch(1)
batch_img, batch_mask = next(training_gen)

start = time.time()
pred_all = model.predict(batch_img)
end = time.time()
print np.shape(pred_all)
print "Detection took " + str(end - start) + " seconds"

for i in range(len(batch_img)):

    im = np.array(batch_img[i], dtype=np.uint8)
    im_mask = np.array(255*batch_mask[i], dtype=np.uint8)
    im_pred = np.array(255*pred_all[i], dtype=np.uint8)

    rgb_mask_pred = cv2.cvtColor(im_pred, cv2.COLOR_GRAY2RGB)
    rgb_mask_pred[:, :, 1:3] = 0*rgb_mask_pred[:, :, 1:2]
    rgb_mask_true = cv2.cvtColor(im_mask, cv2.COLOR_GRAY2RGB)
    rgb_mask_true[:, :, 0] = 0*rgb_mask_true[:, :, 0]
    rgb_mask_true[:, :, 2] = 0*rgb_mask_true[:, :, 2]

    img_pred = cv2.addWeighted(rgb_mask_pred, 0.5, im, 0.5, 0)
    img_true = cv2.addWeighted(rgb_mask_true, 0.5, im, 0.5, 0)

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(im)
    plt.title('Original image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(img_pred)
    plt.title('Predicted segmentation mask')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(img_true)
    plt.title('Ground truth BB')
    plt.axis('off')
    plt.show()
