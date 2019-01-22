from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

from dataGenerator import DataGenerator
from dataUtils import *
from random import randint

class TestAugmenter(TestCase):

    target_dims = (384, 256)

    input = '../../res/input/'
    backgrounds = '../../res/backgrounds/'

    dataGenerator = DataGenerator(input, backgrounds)

    #----------------------------------------------------------------------------------------------------
    #                                              Test Generator                                       #
    #----------------------------------------------------------------------------------------------------

    # rgb generation
    def test_generatorRGB(self):
        """test python generator"""
        # adjust params in dataConfig.py before running script
        batch_size = 16
        gen = self.dataGenerator.generator(batch_size, normalize=False, grayscale=False)

        # get e.g. first element
        images, corners = next(gen)
        assert(len(images) == batch_size)
        image, corner = images[0], corners[0]

        # unwarp
        pts_src = np.reshape(corner, (4, 2))
        # retrieve image height and width
        h, w, _ = image.shape

        pts_dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype = 'float32')
        dewarped_image = warp_image(image, pts_src, pts_dst)

        # visualize
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[1].imshow(cv2.cvtColor(dewarped_image, cv2.COLOR_BGR2RGB))
        plt.show()

    '''
    # grayscale generation
    def test_generatorGray(self):
        """test python generator"""
        # adjust params in dataConfig.py before running script
        batch_size = 16
        gen = self.dataGenerator.generator(batch_size, normalize=False, grayscale=True)

        # get e.g. first element
        images, corners = next(gen)
        image, corner = images[0], corners[0]

        # reshape (just for visualization)
        h, w, _ = image.shape
        image = np.reshape(image, (h, w))

        # unwarp
        pts_src = np.reshape(corner, (4, 2))
        # retrieve image height and width
        h, w = image.shape

        pts_dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype = 'float32')
        dewarped_image = warp_image(image, pts_src, pts_dst)

        # visualize
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(image, cmap='gray')
        ax[1].imshow(dewarped_image, cmap='gray')
        plt.show()
    '''
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
        warped_image, y = self.dataGenerator.augmentSample(img=img, mode=mode, background=random_background)

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
        warped_image, y = self.dataGenerator.augmentPerspective(img, mode=0)
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

        random_patch = self.dataGenerator.augmentBackground(random_background)
        random_patch = cv2.cvtColor(random_patch,  cv2.COLOR_BGR2BGRA)

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

        aug_image = self.dataGenerator.augmentPhotometric(img)


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
        random_patch = self.dataGenerator.augmentBackground(random_background)
        random_patch = cv2.cvtColor(random_patch,  cv2.COLOR_BGR2BGRA)

        # perspective transformation
        img = cv2.imread(self.input + random_img_nm)
        img = cv2.resize(img, (self.target_dims[1], self.target_dims[0]))
        height, width, _ = img.shape
        warped_image, y = self.dataGenerator.augmentPerspective(img, randint(0, 1))

        # boolean mask of alpha channel -> replace with background patch
        idx_transparent = warped_image == [0, 0, 0, 0]
        warped_image[idx_transparent] = random_patch[idx_transparent]

        fig, ax = plt.subplots(nrows=1, ncols=3)

        ax[0].imshow(random_patch)
        ax[1].imshow(img)
        ax[2].imshow(warped_image)
        plt.show()
    '''