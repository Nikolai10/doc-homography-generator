# DocHomographyGenerator (Python2.7)

Keywords: `OCR`, `Page Dewarping`, `Synthetic Dataset Generation`, `Deep Learning` 

## Synthetic Dataset Generation

This work is based on [Recovering Homography from Camera Captured Documents using 
Convolutional Neural Networks (2017)](https://arxiv.org/abs/1709.03524) and aims to provide a synthetic dataset producer/ generator (using various data augmentation methods), which allow to estimate the corner displacement vectors of the distorted document image.

distorted document images:
<p align="center">
  <img width="90%" height="90%" src=/doc/DocHomography.png>
</p>

dewarped document images (first row):
<p align="center">
  <img width="90%" height="90%" src=/doc/dewarpedDocImages.png>
</p>

## 1. Introduction

Capturing document images is a common way for digitizing physical documents due to the ubiquitousness of smartphones. 
In contrast to scans from a flatbed scanner, camera captured documents require a more sophisticated processing pipeline, because of perspective distorted images. In order to restore (dewarp) the original document image, one computes the Homography (3x3 matrix), that maps the corner points of the document image to its canonical position. However, estimating the params of the Homography matrix directly from one single input image is difficult, [see](https://arxiv.org/abs/1709.03524). 

An alternative way of computing H, is the 4pts method (see chapter findHomography [Camera Calibration and 3D Reconstruction](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html)). Having 4 corresponding coplanar points, the distorted image can be unwarped the following way:

        #  Calculate Homography
        h, status = cv2.findHomography(pts_src, pts_dst)
    
        #  Warp source image to destination
        return cv2.warpPerspective(src=img, M=h, dsize=(width, height)) 

## 2. Methodology

The following figures demonstrate the generation process of some sample images.

<p align="center">
  <img width="80%" height="80%" src=/doc/dataAugmPipe.png>
</p>

<p align="center">
  <img width="80%" height="80%" src=/doc/dataAugmPipe2.png>
</p>

**Note**: 
- In contrast to [the original work](https://arxiv.org/abs/1709.03524), this module also allows to generate images 
where corners are outside the image boundaries. 
- The param mode_p (src/config) determines the ratio between included and excluded corners. By default, at least 70% of 
all generated images will only have included corners.
- Dataset Format (stored as .mat file):
``` 
    X = (N, height, width, 3)  
    Y = (N, 8)                  # 4 * x,y (top left, top right, ...,bottom left)
``` 
- Beside the possibility of generating an arbitrarily large dataset (see DataProducer), DocHomography Generator also allows to be used as Python generator (see DataGenerator), where data is only generated batch-by-batch. This is in particular useful, when the dataset is too big to fit into memory (Big Data). For example, in order to train a model using a python generator, one can use the [fit_generator()](https://keras.io/models/sequential/)-method provided by Keras.

## 3. Setup
1. Download textures or background images (e.g. from [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) or 
[MIT Indoor scenes dataset](http://web.mit.edu/torralba/www/indoor.html)) into res/backgrounds as collection of images 
(remove intermediate folders).

2. Insert Pdf images (as PNG) into /res/input as collection of images; **Note:** to convert PDFs to PNGs, one can use 
the scripts provided in /src/data_utils.py 

3. Install dependencies (using pip)
``` 
    pip install -r requirements.txt
``` 

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
