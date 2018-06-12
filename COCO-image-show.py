import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from pycocotools.coco import COCO
import os
from PIL import Image
from PIL import ImageDraw
import csv
pylab.rcParams['figure.figsize'] = (8.0, 10.0)    # 修改默认更新图表大小

# initialize COCO api for person keypoints annotations
dataDir = '/home/devin/Project/coco'
dataType = 'val2017'
# instancesAnnFile = '{}/annotations/annotations_trainval2017/annotations/instances_{}.json'.format(dataDir, dataType)
# keypointsannFile = '{}/annotations/annotations_trainval2017/annotations/person_keypoints_{}.json'.format(dataDir, dataType)
captionsAnnFile = '{}/annotations/annotations_trainval2017/annotations/captions_{}.json'.format(dataDir, dataType)

coco_kps = COCO(captionsAnnFile)

imgIds = coco_kps.getImgIds()

# display COCO categories and supercategories
# cats = coco_kps.loadCats(coco_kps.getCatIds())
# nms=[cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))
# nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories, select one at random
# catIds = coco_kps.getCatIds(catNms=['person']);
# imgIds = coco_kps.getImgIds(catIds=catIds);
# print('there are %d images containing human'%len(imgIds))

for i in range(len(imgIds)):
    imageNameTemp = coco_kps.loadImgs(imgIds[i])[0]
    imageName = imageNameTemp['file_name'].encode('raw_unicode_escape')
    img = coco_kps.loadImgs(imgIds[i])[0]
    I = io.imread('%s/images/%s/%s' % (dataDir, dataType, img['file_name']))
    plt.imshow(I)
    plt.axis('off')
    ax = plt.gca()
    # annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    annIds = coco_kps.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco_kps.loadAnns(annIds)
    coco_kps.showAnns(anns)
    plt.show()

