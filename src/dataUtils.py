from __future__ import division
import cv2
import os


def warp_image(img, pts_src, pts_dst):

    # try rgb
    try:
        height, width, _ = img.shape
    # try grayscale
    except ValueError:
        height, width = img.shape

    #  Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)

    #  Warp source image to destination based on homography; format (4,2)
    return cv2.warpPerspective(src=img, M=h, dsize=(width, height))
