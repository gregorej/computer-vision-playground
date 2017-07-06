from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

dataDir = os.environ['DATASETS'] + '/coco'
dataType = 'train2014'
annFile = '%s/annotations/instances_%s.json' % (dataDir, dataType)
local_image_file_pattern = '%s/images/%d.jpg'


def get_image(img_id):
    local_image_path = local_image_file_pattern % (dataDir, img_id)
    try:
        return io.imread(local_image_path)
    except IOError:
        I = io.imread('http://mscoco.org/images/%d' % (img_id))
        io.imsave(local_image_path, I)
        return I


# initialize COCO api for instance annotations
coco = COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print 'COCO categories: \n\n', ' '.join(nms)

nms = set([cat['supercategory'] for cat in cats])
print 'COCO supercategories: \n', ' '.join(nms)

catIds = coco.getCatIds(catNms=['person','dog','skateboard'])
imgIds = coco.getImgIds(catIds=catIds)
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
img_id = img['id']
I = get_image(img_id)
plt.figure(); plt.axis('off')
plt.imshow(I)
plt.show()

plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img_id, catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)


class CocoDataset(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        dataType = 'train2014'
        annFile = '%s/annotations/instances_%s.json' % (dataDir, dataType)
        self._coco = COCO(annFile)
        category_ids = coco.getCatIds(catNms=['car'])
        self._img_ids = self._coco.getImgIds(catIds=category_ids)

    def _get_image(self, img_id):
        local_image_path = local_image_file_pattern % (self.data_dir, img_id)
        try:
            return io.imread(local_image_path)
        except IOError:
            image = io.imread('http://mscoco.org/images/%d' % (img_id))
            io.imsave(local_image_path, image)
            return image

