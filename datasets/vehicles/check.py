from VehiclesMaskDataset import VehiclesMaskDataset as vehicles
import cv2
import matplotlib.pyplot as plt
import numpy as np
from predefined import crowdai, custom_ds, object_detect



all = vehicles.concat([crowdai[0:2], object_detect[0:2], custom_ds[0:2]])
ds = all

for batch_image, batch_mask in ds.generator(1):
    image = np.array(batch_image[0], dtype=np.uint8)
    im_mask = np.array(255*batch_mask[0], dtype=np.uint8)

    rgb_mask_true = cv2.cvtColor(im_mask, cv2.COLOR_GRAY2RGB)
    rgb_mask_true[:, :, 0] = 0*rgb_mask_true[:, :, 0]
    rgb_mask_true[:, :, 2] = 0*rgb_mask_true[:, :, 2]

    img_true = cv2.addWeighted(rgb_mask_true, 0.5, image, 0.5, 0)

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(im_mask[:,:,0])
    plt.title('Mask BB')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_true)
    plt.title('Ground truth BB')
    plt.axis('off')
    plt.show()
    cv2.waitKey()
