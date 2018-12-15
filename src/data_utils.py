from __future__ import division
from pdf2image import convert_from_path
import cv2
#import textract
import os


def warp_image(img, pts_src, pts_dst):
    height, width, _ = img.shape

    #  Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)

    #  Warp source image to destination based on homography; format (4,2)
    return cv2.warpPerspective(src=img, M=h, dsize=(width, height))

'''
def retrieveGroundTruth(file, input_dir, output_dir):
    """
    retrieve textual content from PFD (as Ground Truth for OCR)

    :param file:                PDF file
    :param input_dir:           dir of PDF file
    :param output_dir:          dir where to store Ground Truth
    :return:
    """

    # output file
    file_out = output_dir + os.path.splitext(file)[0] + '.txt'

    # write to file
    with open(file_out, 'w') as f:
        print(input_dir + file)
        temp = textract.process(input_dir + file)
        print(temp)
        f.write(textract.process(file))
'''

def pdfToPng(input_dir, output_dir):
    """
    for a given input_dir, convert all pdfs to pngs and store in output_dir

    :param input_dir:           dir containing pdfs
    :param output_dir:          dir where to store converted data
    :return:
    """

    count_all_pdfs = 0

    for file_name in os.listdir(input_dir):
        if file_name.split(".")[-1].lower() in "pdf":

            try:
                images = convert_from_path(input_dir + file_name, thread_count=4)
                for idx, image in enumerate(images):
                    image.save(output_dir + file_name.split(".")[0] + "-" + str(idx)  + ".png", "png")

            except:
                print("pdf: {} could not be processed".format(file_name))

    print('Amount of pdfs: {}'.format(count_all_pdfs))