from __future__ import division

import cv2
import numpy as np
from random import randint
import math
import random
from imgaug import augmenters as iaa
from dataConfig import DataConfig
import os
import matplotlib.pyplot as plt

from dataUtils import warp_image


class DataGenerator:
    """

    Synthetic Dataset Generation (for recovering homography and measure overall OCR-performance)


    this class follows the approach described in "Recovering Homography from Camera Captured Documents using
    Convolutional Neural Networks (2017)" to synthesize an evaluation dataset.

    optimized for using keras: fit_generator()
    ref: https://keras.io/models/sequential/

    """

    def __init__(self, input, backgrounds, target_dims = DataConfig.target_dims, shift=DataConfig.shift, p=DataConfig.mode_p):

        """
        Init params

        :param input:           directory of document images (as pdf)
        :param backgrounds:     directory containing background images (e.g. DTD dataset)
        :param target_dims:     image dimension of generated sample
        :param shift:           max abs corner displacement (4pts form a homography)
        :param p:               high p -> high chance for no outliers (perspective distortion)
        """

        self.input = input
        self.backgrounds = backgrounds
        self.target_dims = target_dims
        self.shift = shift
        self.p = p

    def generator(self, batch_size, normalize=True, grayscale=False):
        """
        gernerate training samples using yield (memory efficient implementation)


        :param batch_size:      training samples to generate in one batch
        :param normalize:       if True, divide image by 255 -> [0, 1]
        :return:
        """
        # get name list of document images and backgrounds
        doc_imgs = os.listdir(self.input)
        backgrounds = os.listdir(self.backgrounds)

        h, w = self.target_dims

        # generate empty arrays
        if grayscale:
            batch_X = np.zeros((batch_size, h, w, 1), dtype=np.uint8)
        else:
            batch_X = np.zeros((batch_size, h, w, 3), dtype=np.uint8)

        batch_Y = np.zeros((batch_size, 8))

        while True:
            for i in range(batch_size):

                # retrieve random document image + background
                random_img_nm = doc_imgs[randint(0, len(doc_imgs)-1)]
                random_background = backgrounds[randint(0, len(backgrounds)-1)]

                # read document image and resize to target size (might involve stretching)
                img = cv2.imread(self.input + random_img_nm)
                img = cv2.resize(img, (self.target_dims[1], self.target_dims[0]))
                height, width, _ = img.shape

                # augment sample (p % for mode 0)
                mode = np.random.choice(np.array([0, 1]), p=[self.p,1-self.p])
                warped_image, y = self.augmentSample(img=img, mode=mode, background=random_background)

                # produce grayscale image
                if grayscale:
                    warped_image = cv2.cvtColor(warped_image,  cv2.COLOR_BGR2GRAY)
                    warped_image = np.reshape(warped_image, (h, w, 1))

                # normalize (more sophisticated methods can be added later)
                if normalize:
                    warped_image = warped_image/np.array(255.0).astype(np.float16) # minimize memory consumption

                # store
                batch_X[i] = warped_image
                batch_Y[i] = y

            # produce next batch
            yield batch_X, batch_Y

    def augmentSample(self, img, mode, background):
        """
        augment one sample image

        :param img:
        :param mode:
        :param background:
        :return:
        """
        # perspective transformation
        warped_image, y = self.augmentPerspective(img, mode)

        # background transformation
        background = self.augmentBackground(background)

        # boolean mask of alpha channel -> replace with background patch
        idx_transparent = warped_image == [0, 0, 0, 0]
        warped_image[idx_transparent] = background[idx_transparent]

        # photometric transformation
        warped_image = self.augmentPhotometric(warped_image)
        warped_image = cv2.cvtColor(warped_image,  cv2.COLOR_BGRA2BGR)
        return warped_image, y

    def augmentPerspective(self, img, mode):
        """
        Perspective Transformation using 4 pts

        :param img:
        :param mode:        mode == 0: no outliers
        :return:
        """
        img = cv2.cvtColor(img,  cv2.COLOR_BGR2BGRA)
        height, width, _ = img.shape

        # generate random offsets
        y = self.random_offset(height, width, mode)

        pts_src = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype = 'float32')
        pts_dst = np.array(np.reshape(y, (4, 2)), dtype= 'float32') # lt, rt, rb, lb

        warped_image = warp_image(img, pts_src, pts_dst)
        return warped_image, y

    def augmentBackground(self, random_background):
        """
        Add randomly sampled textures;
        in this work: MIT Indoor scenes dataset             DTD
                                       - office             - banded
                                       - meeting room       - checkered
                                       - waiting room       - knitted

        gifs removed: vestibul2_gif.jpg; conferencerm2_gif.jpg
        Total unique image: 853 (how to extend -> simply add raw images to res/backgrounds)

        :param random_background:     random_background image
        :return:
        """
        # read random image
        background = cv2.imread(self.backgrounds + random_background)

        # preprocessing
        background = self.scale_image(background)
        background = cv2.cvtColor(background,  cv2.COLOR_BGR2BGRA)

        # compute random crop
        h, w, _ = background.shape
        target_h, target_w = self.target_dims
        x_max, y_max = w - target_w, h - target_h
        rand_x, rand_y = randint(0, x_max), randint(0, y_max)

        # generate random crop of background image
        random_crop = background[rand_y:rand_y+target_h,rand_x:rand_x + target_w]
        assert(random_crop.shape[0:2] == self.target_dims)
        return random_crop

    def adjust_gamma(self, image, gamma=1.0):
        """
        Gamma correction using:
        https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/

        :param image:
        :param gamma:
        :return:
        """
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype('uint8')

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def augmentPhotometric(self, image, sigma=DataConfig.sigma, scale=DataConfig.scale, gamma=DataConfig.gamma):
        """
        add photometric augmentations to image

        [original paper:]
        add illumination variations and different noises (such as motion and defocus blur)
        - add motion blur of variable angles and magnitudes to the resultant images to simulate camera movements
        - add Gaussian blur to the images to model dirty lenses and defocus blur.
        - lighting variations: create different filters based on gamma transformations in spatial domain

        [in this work:]
        gaussian blur, additive gaussian noise and gamma correction (randomly applied)

        :param image:
        :return:
        """

        if(random.randint(0, 1)==1):
            aug_blur = iaa.Sequential([iaa.GaussianBlur(sigma=sigma)])
            image = aug_blur.augment_image(image)

        if(random.randint(0, 1)==1):
            aug_noise = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=scale*255)])
            image = aug_noise.augment_image(image)

        if(random.randint(0, 1)==1):
            image = self.adjust_gamma(image, round(random.uniform(gamma[0], gamma[1]), 2))

        return image

    def scale_image(self, background):
        """
        helper: scale background to bigger size if required

        :param background:
        :return:
        """
        height, width = self.target_dims
        h, w, _ = background.shape

        if h < height or w < width:
            y_fac = height/h
            x_fac = width/width
            scalar = math.ceil(max(y_fac, x_fac))
            background = cv2.resize(background, (0, 0), fx=scalar, fy=scalar)

        return background

    def random_offset(self, height, width, mode=0):
        """
        helper: generate random offset for each corner

        :param height:
        :param width:
        :param mode:            mode == 0: no outlier
        :return:
        """

        if mode == 0:
            # top corner left and right
            tl_x, tl_y = 0 + randint(0, self.shift), 0 + randint(0, self.shift)
            tr_x, tr_y = width - randint(0, self.shift), 0 + randint(0, self.shift)
            # bottom corner left and right
            br_x, br_y = width - randint(0, self.shift), height - randint(0, self.shift)
            bl_x, bl_y = 0 + randint(0, self.shift), height - randint(0, self.shift)
        else:
            tl_x, tl_y = 0 + randint(-self.shift, self.shift), 0 + randint(-self.shift, self.shift)
            tr_x, tr_y = width + randint(-self.shift, self.shift), 0 + randint(-self.shift, self.shift)

            br_x, br_y = width + randint(-self.shift, self.shift), height + randint(-self.shift, self.shift)
            bl_x, bl_y = 0 + randint(-self.shift, self.shift), height + randint(-self.shift, self.shift)

        return np.array([tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y])