import matplotlib.pyplot as plt
from vehicles_dataset import get_sample_by_index, img_size
from plot_utils import plot_im_bbox, plot_bbox

sample = get_sample_by_index(1)
sample.resize(img_size)
img_mask = sample.merged_mask()

img = sample.image

dst = sample.copy()
dst.stretch(80)

bb_boxes = dst.bb_boxes
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.imshow(img)
# plt.plot(p1[0], p1[1], 'mo')
# plt.plot(p2[0], p2[1], 'mo')
# plt.plot(p3[0], p3[1], 'mo')
# plt.plot(p4[0], p4[1], 'mo')
for i in range(len(bb_boxes)):
    plot_bbox(bb_boxes, i, 'g')

    bb_box_i = [bb_boxes.iloc[i]['xmin'], bb_boxes.iloc[i]['ymin'],
                bb_boxes.iloc[i]['xmax'], bb_boxes.iloc[i]['ymax']]
    plt.plot(bb_box_i[0], bb_box_i[1], 'rs')
    plt.plot(bb_box_i[2], bb_box_i[3], 'bs')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(dst.image)
# bb_boxes1 = bb_boxes.copy(deep=True)
# bb_boxes1['xmin'] = (bb_boxes['xmin'] - p1[0])/(p2[0]-p1[0])*img.shape[1]
# bb_boxes1['xmax'] = (bb_boxes['xmax'] - p1[0])/(p2[0]-p1[0])*img.shape[1]
# bb_boxes1['ymin'] = (bb_boxes['ymin'] - p1[1])/(p3[1]-p1[1])*img.shape[0]
# bb_boxes1['ymax'] = (bb_boxes['ymax'] - p1[1])/(p3[1]-p1[1])*img.shape[0]
plt.plot(0, 0, 'mo')
plt.plot(img.shape[1], 0, 'mo')
plt.plot(img.shape[1], img.shape[0], 'mo')
plt.plot(0, img.shape[0], 'mo')
plot_im_bbox(dst)

plt.axis('off')

sample = get_sample_by_index(1)
img_mask = sample.merged_mask()

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plot_im_bbox(sample)

plt.subplot(2, 2, 2)
plt.imshow(img_mask[:, :, 0])
plt.axis('off')

plt.subplot(2, 2, 3)
# dst, bb_boxes1 = trans_image(img, bb_boxes, 100)
# dst, bb_boxes1 = stretch_image(img, bb_boxes, 100)
dst = sample.copy()
dst.stretch(100)

plt.imshow(dst.image)

plot_im_bbox(dst)

plt.subplot(2, 2, 4)
img_mask2 = dst.merged_mask()
plt.imshow(img_mask2[:,:,0])
plt.axis('off')

plt.show()
