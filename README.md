# DocHomographyGenerator (Python2.7)

Keywords: `OCR`, `Page Dewarping`, `Synthetic Dataset Generation`, `Deep Learning` 

## Synthetic Dataset Generation

based on [Recovering Homography from Camera Captured Documents using 
Convolutional Neural Networks (2017)](https://arxiv.org/abs/1709.03524)

distorted document images:
![Application Overview](/doc/DocHomography.png)

dewarped document images (first row):
![Application Overview INV](/doc/dewarpedDocImages.png)

## 1. Introduction

Capturing document images is a common way for digitizing physical documents due to the ubiquitousness of smartphones. 
In contrast to a flatbed scanner, camera captured documents require a more sophisticated processing pipeline, because of
perspective distorted images (among others). To restore the original document image, one computes the Homography 
(3x3 matrix), that maps the points in one image to the corresponding points in the other image (or to its canonical 
positions). However, estimating the params of the Homography matrix directly from one single input image is difficult, 
[see](https://arxiv.org/abs/1709.03524). An alternative way of computing H, is the 4pts method (see chapter findHomography 
[Camera Calibration and 3D Reconstruction](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html)).

This work aims to provide a synthetic dataset (using various data augmentation methods), which allow to estimate the 
corner displacement vectors of the distorted document image. Having 4 corresponding coplanar points, the distorted image 
can be unwarped the following way:

        #  Calculate Homography
        h, status = cv2.findHomography(pts_src, pts_dst)
    
        #  Warp source image to destination
        return cv2.warpPerspective(src=img, M=h, dsize=(width, height)) 

## 2. Methodology

Generating one training sample consists of the following steps:
- load random image
- augment perspective distortion (the 4 corner points are randomly shifted by max e.g. 100 pixels, followed by computing H)
- generate random background patch (e.g. texture image, random crop)
- data augmentations (gaussian blur, additive gaussian noise and gamma correction as illumination variations)

**Note**: 
- In contrast to [the original work](https://arxiv.org/abs/1709.03524), this module also allows to generate images, 
where corners are outside the image boundaries. 
- The param mode_p (src/config) determines the ratio between included and excluded corners. By default, at least 70% of 
all generated images will only have included corners.

![Example_1](/doc/dataAugmPipe.png)

![Example_2](/doc/dataAugmPipe2.png)

Dataset Format (stored as .mat file):
    
    X = (N, height, width, 3)  
    Y = (N, 8)                  # 4 * x,y (top left, top right, ...,bottom left)
    
## 3. Setup

Install dependencies

    pip install -r requirements.txt
    
1. Download textures or background images (e.g. from [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) or 
[MIT Indoor scenes dataset](http://web.mit.edu/torralba/www/indoor.html)) into res/backgrounds as collection of images 
(remove intermediate folders).

2. Insert Pdf images (as PNG) into /res/input as collection of images; **Note:** to convert PDFs to PNGs, one can use 
the scripts provided in /src/data_utils.py 

## 4. File Structure

    res                               
        ├── backgrounds                 # background images (gif not supported)              
        ├── input                       # pdf images 
        ├── output                      # location where to store generated dataset + corners as .mat file
    src
        ├── unit_tests                  # unit tests demonstrating functionality
            ├── ...
        ├── dataGenerator.py            # Data Synthesis optimized for Keras fit_generator()
        ├── dataProducer.py             # Data Synthesis using multiprocessing (fixed set)
        ├── dataConfig.py               # all config params of data synthesis
        ├── dataUtils.py                # helper methods       
    requirements.txt                    # dependencies (Python2.7) 

## 5. Usage
    
    # init Augmenter with input, output and backgrounds
    augmenter = AugmenterV2(input, output, backgrounds)
    
    # e.g. generate 100 document images
    augmenter.augmentDataset_master(max=100, mode_p=0.7)

For more information: see src/unit_tests

## License

![DocHomographyGenerator_license](LICENSE)
