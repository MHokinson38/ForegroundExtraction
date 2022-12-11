#############################
# Unit tests for graphCut.py
#
# Matthew Hokinson, 12/10/22
#############################
import pytest 
import cv2
import matplotlib.pyplot as plt

from ExtractionMethods.graphCut import GraphCut 

# Testing globals 
TEST_IMAGE_DIR = "ExampleImages/"

# Test helping methods 
def read_image(path):
    """
    Read in the grayscale image at the given path
    """
    return cv2.imread(TEST_IMAGE_DIR + path, cv2.IMREAD_GRAYSCALE)

def test_tiny_cut():
    """
    Test the GraphCut class on the tiny.png image
    """
    # Read in the image
    img = read_image("tiny.png")

    # Create the seeds for the image 
    foregroundSeeds = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    backgroundSeeds = [(6, 0), (6, 1), (6, 2), (6, 3), (7, 0), (7, 1), (7, 2), (7, 3)]

    gc = GraphCut(img, foregroundSeeds, backgroundSeeds)
    foreground_img = gc.extract_foreground()

    # Verify the foreground image is properly selected 
    for i in range(4):
        for j in range(img.shape[1]):
            assert foreground_img[i, j] == img[i, j]     

    cv2.imwrite("tiny_foreground.png", foreground_img)  

def test_logo_small_cut():
    """
    Test the GraphCut class on the logo
    """
    # Read in the image
    img = read_image("logo_small.png")

    # Create the seeds for the image 
    foregroundSeeds = [(14, 14), (14, 15), (14, 15), (14, 17), 
                        (15, 14), (15, 15), (15, 16), (15, 17), 
                        (16, 14), (16, 15), (16, 16), (16, 17), 
                        (17, 14), (17, 15), (17, 16), (17, 17)]
    backgroundSeeds = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]

    gc = GraphCut(img, foregroundSeeds, backgroundSeeds)
    foreground_img = gc.extract_foreground()

    # Verify the foreground image is only the non-zero pixels of the original image 
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if foreground_img[i, j] != 0:
                assert foreground_img[i, j] == img[i, j]
            else:
                assert foreground_img[i, j] == 0