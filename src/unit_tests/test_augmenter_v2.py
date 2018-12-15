from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import cv2
from random import randint
import os

from data_utils import warp_image
from src.augmenter_v2 import AugmenterV2


class TestAugmenter(TestCase):

    target_dims = (384, 256)

    input = '../../res/input/'
    output = '../../res/output/'
    backgrounds = '../../res/backgrounds/'

    augmenter = AugmenterV2(input, output, backgrounds)

    #----------------------------------------------------------------------------------------------------
    #                                           Generate 100 samples                                    #
    #----------------------------------------------------------------------------------------------------

    def test_augmentDataset(self):
        """generate dataset"""
        self.augmenter.augmentDataset_master(max=10000, mode_p=0.7)

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

    '''
    def test_augmentPerspective(self):

        # get name list of document images and backgrounds
        doc_imgs = os.listdir(self.input)

        # retrieve random document image + background
        random_img_nm = doc_imgs[randint(0, len(doc_imgs)-1)]

        # read document image and resize to target size (might involve strechting)
        img = cv2.imread(self.input + random_img_nm)
        img = cv2.resize(img, (self.target_dims[1], self.target_dims[0]))
        height, width, _ = img.shape

        fig, ax = plt.subplots(nrows=1, ncols=3)
        warped_image, y = self.augmenter.augmentPerspective(img, mode=0)
        print(y)
        ax[0].imshow(img)
        ax[1].imshow(warped_image)

        # unwarp
        pts_src = np.reshape(y, (4, 2))
        pts_dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype = 'float32')

        dewarped_image = warp_image(warped_image, pts_src, pts_dst)
        ax[2].imshow(dewarped_image)

        plt.show()
    '''

    '''
    def test_augmentBackground(self):
        """generate random background with target dims"""
       
        backgrounds = os.listdir(self.backgrounds)
        random_background = backgrounds[randint(0, len(backgrounds)-1)]

        random_patch = self.augmenter.augmentBackground(random_background)
        random_patch = cv2.cvtColor(random_patch,  cv2.COLOR_RGB2RGBA)

        plt.imshow(random_patch)
        plt.show()
    '''

    '''
    def test_augmentPhotometric(self):

        # get name list of document images
        doc_imgs = os.listdir(self.input)

        # retrieve random document image
        random_img_nm = doc_imgs[randint(0, len(doc_imgs)-1)]

        # read document image and resize to target size (might involve strechting)
        img = cv2.imread(self.input + random_img_nm)
        img = cv2.resize(img, (self.target_dims[1], self.target_dims[0]))
        height, width, _ = img.shape

        aug_image = self.augmenter.augmentPhotometric(img)


        fig, ax = plt.subplots(nrows=1, ncols=2)

        ax[0].imshow(img)
        ax[1].imshow(aug_image)
        plt.show()
    '''

    '''
    def test_augmentPerspectiveANDBackground(self):

        backgrounds = os.listdir(self.backgrounds)
        doc_imgs = os.listdir(self.input)
        random_background = backgrounds[randint(0, len(backgrounds)-1)]
        random_img_nm = doc_imgs[randint(0, len(doc_imgs)-1)]

        # generate random background
        random_patch = self.augmenter.augmentBackground(random_background)
        random_patch = cv2.cvtColor(random_patch,  cv2.COLOR_RGB2RGBA)

        # perspective transformation
        img = cv2.imread(self.input + random_img_nm)
        img = cv2.resize(img, (self.target_dims[1], self.target_dims[0]))
        height, width, _ = img.shape
        warped_image, y = self.augmenter.augmentPerspective(img, randint(0, 1))

        # boolean mask of alpha channel -> replace with background patch
        idx_transparent = warped_image == [0, 0, 0, 0]
        warped_image[idx_transparent] = random_patch[idx_transparent]

        fig, ax = plt.subplots(nrows=1, ncols=3)

        ax[0].imshow(random_patch)
        ax[1].imshow(img)
        ax[2].imshow(warped_image)
        plt.show()
    '''