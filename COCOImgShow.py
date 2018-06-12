import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from pycocotools.coco import COCO
import os
from PIL import Image
from PIL import ImageDraw
import csv


# display COCO categories and supercategories
def displaycats(coco_kps):

    cats = coco_kps.loadCats(coco_kps.getCatIds())
    nms = [cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))
    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))
    # get all images containing given categories, select one at random
    catIds = coco_kps.getCatIds(catNms=['person']);
    imgIds = coco_kps.getImgIds(catIds=catIds);
    print('there are %d images containing human' % len(imgIds))

    return catIds, imgIds


def showImage(imgId):

    imageNameTemp = coco_kps.loadImgs(imgId)[0]
    imageName = imageNameTemp['file_name'].encode('raw_unicode_escape')
    img = coco_kps.loadImgs(imgIds[i])[0]
    I = io.imread('%s/images/%s/%s' % (dataDir, dataType, img['file_name']))
    plt.imshow(I)
    plt.axis('off')
    ax = plt.gca()
    try:
        annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    except Exception as e:
        annIds = coco_kps.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco_kps.loadAnns(annIds)
    coco_kps.showAnns(anns)
    plt.show()



if __name__ == "__main__":

    pylab.rcParams['figure.figsize'] = (8.0, 10.0)  # 修改默认更新图表大小
    # initialize COCO api for person keypoints annotations
    dataDir = '/home/devin/Project/coco'
    dataType = 'val2017'
    keyPointAnnFile = '{}/annotations/annotations_trainval2017/annotations/person_keypoints_{}.json'.format(dataDir, dataType)
    instanceAnnFile = '{}/annotations/annotations_trainval2017/annotations/instances_{}.json'.format(dataDir, dataType)
    captionsAnnFile = '{}/annotations/annotations_trainval2017/annotations/captions_{}.json'.format(dataDir, dataType)
    coco_kps = COCO(captionsAnnFile)
    imgIds = []

    try:
        catIds, imgIds = displaycats(coco_kps)
        # print(type(imgIds))
    except Exception as e:
        print(e)

    if len(imgIds) == 0:

        imgIds = coco_kps.getImgIds()

    if len(imgIds) > 0:

        for i in range(len(imgIds)):

            showImage(imgIds[i])