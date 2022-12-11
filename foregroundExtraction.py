#############################
# Various implementations of Foreground Extraction on a given image 
#
# Matthew Hokinson, 12/10/22
#############################

import cv2 
import numpy as np

from ExtractionMethods.graphCut import GraphCut

import argparse
parser = argparse.ArgumentParser(description='Extract the foreground of an image')
parser.add_argument('image_path', type=str, help='The path to the image to extract the foreground from')
parser.add_argument('-v', '--verbose', action='store_true', help='Print out debugging information')
parser.add_argument('-r', '--radius', type=int, help='The radius of the selection circle')

DEBUG = False
def debug(str):
    if DEBUG:
        print(str)

SELECTION_RADIUS = 25

def add_seeds(img, seeds, x, y):
    """
    Add the seeds to the image 
    """
    for i in range(-SELECTION_RADIUS, SELECTION_RADIUS):
        for j in range(-SELECTION_RADIUS, SELECTION_RADIUS):
            # if point is outside the radius, skip 
            if i**2 + j**2 > SELECTION_RADIUS**2:
                continue

            if x + i >= 0 and x + i < img.shape[0] and y + j >= 0 and y + j < img.shape[1]:
                seeds.append((x + i, y + j))

def on_mouse_event(event, x, y, flags, param):
    # left button is pressed +
    # mouse moves
    if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON):
        debug('click L, ' + str([y, x]))
        add_seeds(img, foregroundSeeds, y, x)
        cv2.circle(display_img, (x, y), SELECTION_RADIUS, (255, 255, 255), -1)

    # when shift is pressed +
    # left button is pressed +
    # mouse moves
    if event == cv2.EVENT_RBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_RBUTTON):
        debug('click R, ' + str([y, x]))
        add_seeds(img, backgroundSeeds, y, x)
        cv2.circle(display_img, (x, y), SELECTION_RADIUS, (0, 0, 0), -1)

# Seeds for the image 
foregroundSeeds = []
backgroundSeeds = []

if __name__ == '__main__':
    # Parse the command line arguments for the image path 
    args = parser.parse_args()
    image_path = args.image_path
    if args.verbose:
        print("Verbose mode enabled")
        DEBUG = True

    if args.radius:
        SELECTION_RADIUS = args.radius

    # read the grayscale image 
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    display_img = img.copy()

    # begin the openCv window 
    root_win = 'Set Foreground and Background Seeds'
    cv2.namedWindow(root_win)

    cv2.setMouseCallback(root_win, on_mouse_event)

    while True: 
        cv2.imshow(root_win, display_img)
        # Wait for a key press 
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

        # if the key is enter, then extract the foreground
        if key == 13:
            debug("Begining the foreground cut")
            debug(f"Foreground seeds: {foregroundSeeds}")
            debug(f"Background seeds: {backgroundSeeds}")
            # create the graph cut object 
            gc = GraphCut(img, foregroundSeeds, backgroundSeeds)
            foreground_img = gc.extract_foreground()

            # show the foreground image 
            cv2.imshow('Foreground', foreground_img)

