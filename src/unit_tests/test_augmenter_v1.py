from unittest import TestCase

from augmenter_v1 import AugmenterV1
from data_utils import warp_image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from random import randint
import os


class TestAugmenter(TestCase):

    target_dims = (384, 256)

    input = '../../res/input/'
    output = '../../res/output/'
    backgrounds = '../../res/backgrounds/'

    augmenter = AugmenterV1(input, output, backgrounds)

    #----------------------------------------------------------------------------------------------------
    #                                           Generate 100 samples                                    #
    #----------------------------------------------------------------------------------------------------

    def test_augmentDataset(self):
        self.augmenter.augmentDataset(max=100)

    #----------------------------------------------------------------------------------------------------
    #                                       Uncomment to Test Components                                #
    #----------------------------------------------------------------------------------------------------

    '''
    def test_augmentSample(self, p=0.7):
        """generate random sample"""

        # get name list of document images and backgrounds
        doc_imgs = os.listdir(self.input)
        backgrounds = os.listdir(self.backgrounds)

        # retrieve random document image + background
        random_img_nm = doc_imgs[randint(0, len(doc_imgs)-1)]
        random_background = backgrounds[randint(0, len(backgrounds)-1)]

        # read document image and resize to target size (might involve strechting)
        img = cv2.imread(self.input + random_img_nm)
        img = cv2.resize(img, (self.target_dims[1], self.target_dims[0]))
        height, width, _ = img.shape

        # augment sample (p % for mode 0)
        mode = np.random.choice(np.array([0, 1]), p=[p,1-p])
        warped_image, y = self.augmenter.augmentSample(img=img, mode=mode, background=random_background)

        fig, ax = plt.subplots(nrows=1, ncols=3)

        ax[0].imshow(img)
        ax[1].imshow(warped_image)

        # unwarp
        pts_src = np.reshape(y, (4, 2))
        pts_dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype = 'float32')

        dewarped_image = warp_image(warped_image, pts_src, pts_dst)
        ax[2].imshow(dewarped_image)

        plt.show()
    '''
    #----------------------------------------------------------------------------------------------------
    #                       for more Components-Tests, see test_augmenter_v2.py                         #
    #----------------------------------------------------------------------------------------------------