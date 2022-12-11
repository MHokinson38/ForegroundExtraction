#############################
# Unit tests for imageGraph.py
#
# Matthew Hokinson, 12/10/22
#############################

import pytest 
import cv2

from ExtractionMethods.Utils.imageGraph import ImageGraph

# Testing globals 
TEST_IMAGE_DIR = "ExampleImages/"

def empty_weight_provider(n1, n2, image):
    return 0

def empty_regional_weight_provider(n1, isForeground, image):
    return 0

# Test helping methods 
def read_image(path):
    """
    Read in the grayscale image at the given path
    """
    return cv2.imread(TEST_IMAGE_DIR + path, cv2.IMREAD_GRAYSCALE)

def test_logo_small():
    """
    Test the ImageGraph class on the logo_small.png image
    """
    # Create the ImageGraph
    IMAGE_DIMENSIONS = 32
    img = read_image("logo_small.png")

    # Read in the image 
    imageGraph = ImageGraph(img)
    imageGraph.build_graph(empty_weight_provider)

    # Check the number of nodes
    assert imageGraph.N == IMAGE_DIMENSIONS * IMAGE_DIMENSIONS
    assert len(imageGraph.get_vertices()) == IMAGE_DIMENSIONS * IMAGE_DIMENSIONS

    for node in imageGraph.get_vertices():
        # Verify the edge count is at least 3 (corner) and at most 8 (middle)
        assert len(imageGraph.get_edges(node)) <= 8
        assert len(imageGraph.get_edges(node)) >= 3

        # Check the edge weights 
        for neighbor in imageGraph.get_edges(node):
            assert imageGraph.get_edge_weight(node, neighbor) == 0

    # Arbitrarily pick a node and make some weight modifications 
    node = imageGraph.get_vertices()[0]
    neighbor = imageGraph.get_neighbors(node)[0]

    assert imageGraph.get_max_weight(node) == 0

    imageGraph.set_weight(node, neighbor, 10)

    assert imageGraph.get_edge_weight(node, neighbor) == 10
    assert imageGraph.get_max_weight(node) == 10

    # Check the source and sink
    assert imageGraph.source == None
    assert imageGraph.sink == None

    imageGraph.add_weighted_source_sink(empty_regional_weight_provider)
    
    # All neightbors present 
    assert len(imageGraph.get_neighbors(imageGraph.source)) == IMAGE_DIMENSIONS * IMAGE_DIMENSIONS
    assert len(imageGraph.get_neighbors(imageGraph.sink)) == IMAGE_DIMENSIONS * IMAGE_DIMENSIONS

def test_tiny_labeling():
    IMAGE_DIMENSIONS = 8
    img = read_image("tiny.png")

    # Read in the image 
    imageGraph = ImageGraph(img)
    imageGraph.build_graph(empty_weight_provider)

    foregroundSeeds = []
    for row in range(4):
        for col in range(IMAGE_DIMENSIONS):
            foregroundSeeds.append((row, col))
    imageGraph.set_seeds(foregroundSeeds, [])

    # Get the foreground of the image, verify only the foreground seeds are set 
    foreground = imageGraph.get_foreground()
    for row in range(IMAGE_DIMENSIONS):
        for col in range(IMAGE_DIMENSIONS):
            if row < 4:
                assert foreground[row][col] != 0 # Not Black 
            else:
                assert foreground[row][col] == 0 # Black (masked)

    # Check that the foreground mask is correct 
    foregroundMask = imageGraph.get_foreground_mask()
    for row in range(IMAGE_DIMENSIONS):
        for col in range(IMAGE_DIMENSIONS):
            if row < 4:
                assert foregroundMask[row][col] == 255 
            else:
                assert foregroundMask[row][col] == 0

