from __future__ import annotations
#############################
# Graph Cut Implementation
#
# Matthew Hokinson, 12/7/22
#############################

import numpy as np
import math

from functools import partial

from ExtractionMethods.Utils.imageGraph import ImageGraph
import ExtractionMethods.Utils.minCutAlgorithms as MinCutAlgorithms

'''
- TODO: Need quick way to redo cuts when adding seeds on the fly.
     We could store a Cut object, and just call redoCut() on it (to store the current max flow state).
     Easy-ish to change later
'''

class GraphCut:
    """
    Graph cut implementation for foreground extraction
    """
    LAMBDA = 1.0  # Weight for regional penalty (TODO: Tune this)
    GAMMA = 1.0  # Weight for boundary penalty (TODO: Tune this)
    SIGMA = 30   # Intensity Difference Threshhold, 30 suggested by blog post: https://julie-jiang.github.io/image-segmentation/

    def __init__(self, image: np.ndarray, foregroundSeeds: list[(int, int)], backgroundSeeds: list[(int, int)]):
        """Init Function 

        Args:
            image (WxHx1 numpy array): Numpy array representation of the image 
                - W: Width of the image
                - H: Height of the image
                - 1: Number of channels in the image (1 for greyscale, 3 for RGB), only support greyscale for now
            foregroundSeeds (list[(int, int)]): List of (row, col) tuples for the foreground seeds
            backgroundSeeds (list[(int, int)]): List of (row, col) tuples for the background seeds
        """
        self.foregroundSeeds = foregroundSeeds
        self.backgroundSeeds = backgroundSeeds

        # Verify the image
        if len(image.shape) != 2:
            raise ValueError(
                "Image must be a WxHx1 numpy array (Greyscale only!)")
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        self.imageGraph = ImageGraph(image)

    def __get_regional_penalties(self, image: np.ndarray) -> np.array:
        """
        **Moving to function for clairty**

        Calculate the regional penalty for a given pixel based on foreground and 
        background intensity distributions in the image 

        Args:
            image (np.ndarray): Pixel to calculate the regional penalty for

        Returns:
            np.ndarray: Regional penaly for each pixel value 
        """
        regionalPenalties = np.zeros((2, 256))

        foregroundValues = np.array([image[r, c] for r, c in self.foregroundSeeds])
        backgroundValues = np.array([image[r, c] for r, c in self.backgroundSeeds])

        for i in range(256):
            # Calculate the number of foreground/background pixels with intensity i
            # Pr = Pixels in (F/B) with intensity i / (F/B) pixels
            # Need to add smoothing component to avoid divide by zero
            # Pr = (Pixels in (F/B) with intensity i + alpha) / (F/B) pixels + 2*alpha

            alpha = 1  # This can be tuned
            prIntensityForeground = (np.count_nonzero(
                foregroundValues == i) + alpha) / (len(foregroundValues) + 2*alpha)
            prIntensityBackground = (np.count_nonzero(
                backgroundValues == i) + alpha) / (len(backgroundValues) + 2*alpha)

            # Calculate the regional penalties
            regionalPenalties[0, i] = -math.log(prIntensityForeground)
            regionalPenalties[1, i] = -math.log(prIntensityBackground)

        # print(regionalPenalties)

        return regionalPenalties

    def extract_foreground(self) -> np.ndarray:
        """
        Graph cut implementation for foreground extraction

        Returns:
            np.ndarray: Foreground image 
        """
        # Weight Setting Callbacks
        def __get_boundary_weight(p1: ImageGraph.Node, p2: ImageGraph.Node, image: np.ndarray) -> int:
            intensity1 = int(image[p1.row, p1.col])
            intensity2 = int(image[p2.row, p2.col]) # Cast to int to avoid overflow

            dSquare = (intensity1 - intensity2)**2
            distance = np.sqrt((p1.row - p2.row)**2 + (p1.col - p2.col)**2)
            rawWeight = GraphCut.GAMMA * math.exp(-dSquare / (2 * GraphCut.SIGMA**2)) / distance

            # print(f"Boundary weight for {p1} and {p2}: dsquare:{dSquare}, distance:{distance}, rawWeight:{rawWeight}")

            return int(100 * rawWeight) # Cast to int, preserve 2 sig. decimal digits 

        def __get_regional_weight(boundaryWeightSums: dict[ImageGraph.Node, int],
                                  regionalPenalties: np.ndarray,
                                  pixel: ImageGraph.Node,
                                  isForeground: bool,
                                  image: np.ndarray) -> int:
            # Do general weighting if pixel is not seeded
            if pixel.label == None:
                # Set the penalty to the liklihood of the pixel being in the opposite region 
                # i.e. for node->source, we set the weight as the penalty of node being in the background 
                regionalWeight = GraphCut.LAMBDA * regionalPenalties[1 if isForeground else 0, image[pixel.row, pixel.col]]
                return int(100 * regionalWeight) # Cast to int, preserve 2 sig. decimal digits 

            # Weighting for seeded pixels
            # All (and only) seeded pixels should be in boundary_weight_sums, throw error if not
            if pixel not in boundaryWeightSums:
                raise ValueError(
                    f"Seeded pixel not in boundary_weight_sums, pixel: {pixel}")

            if isForeground:
                if pixel.label == ImageGraph.Label.FOREGROUND:
                    return 1 + boundaryWeightSums[pixel]
                return 0
            else:
                if pixel.label == ImageGraph.Label.BACKGROUND:
                    return 1 + boundaryWeightSums[pixel]
                return 0

        self.imageGraph.set_seeds(
            foregroundSeeds=self.foregroundSeeds, backgroundSeeds=self.backgroundSeeds)
        print(f"Seeded pixels: ", self.imageGraph.get_seeded_pixels())
        self.imageGraph.build_graph(__get_boundary_weight)

        print("Built graph...")

        # Print out boundary weights (look good) 
        # for node in self.imageGraph.get_vertices():
        #     for neighbor in self.imageGraph.get_edges(node):
        #         print(f"Boundary weight for: ({node}) -> ({neighbor}) = {self.imageGraph.get_edge_weight(node, neighbor)}")

        regionalPenalties = self.__get_regional_penalties(self.imageGraph.image)

        # Find all current weight sums for seeded pixels (only boundary weights)
        boundaryWeightSums = {node: self.imageGraph.get_total_weight(node) 
                    for node in self.imageGraph.get_seeded_pixels()}
        self.imageGraph.add_weighted_source_sink(
            partial(__get_regional_weight, boundaryWeightSums, regionalPenalties))

        print("Added source and sink...")

        # Print the weights for source and sink connections 
        # for node in self.imageGraph.get_vertices():
        #     if node == self.imageGraph.source or node == self.imageGraph.sink:
        #         continue
        #     print(f"Source weight for: ({node}) = {self.imageGraph.get_edge_weight(node, self.imageGraph.source)}")
        #     print(f"Sink weight for: ({node}) = {self.imageGraph.get_edge_weight(node, self.imageGraph.sink)}")

        # Perform graph cut using some algorithm (can be swapped for other algos)
        # Does the cut, and updates the labels of the nodes in the graph
        MinCutAlgorithms.pushRelabelCut(self.imageGraph)

        return self.imageGraph.get_foreground()

    def add_foreground_seed(self, row: int, col: int):
        """Add a foreground seed to the graph cut

        Args:
            row (int): Row of the seed
            col (int): Col of the seed
        """
        self.foregroundSeeds.append((row, col))
        self.imageGraph.set_seed(row, col, isForeground=True) 

    def add_background_seed(self, row: int, col: int):
        """Add a background seed to the graph cut

        Args:
            row (int): Row of the seed
            col (int): Col of the seed
        """
        self.backgroundSeeds.append((row, col))
        self.imageGraph.set_seed(row, col, isForeground=False)
