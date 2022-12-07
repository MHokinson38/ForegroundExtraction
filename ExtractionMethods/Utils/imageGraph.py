#############################
# Data Structure for storing images as a weighted graph
# for use in graph cut algorithms
#
# Matthew Hokinson, 12/6/22
#############################

import numpy as np

from __future__ import annotations
from collections.abc import Callable
import enum


class ImageGraph:
    """
    Data structure for storing images as a weighted graph for use in graph cut algorithms

    Graph Representation:
        - Each pixel is a node in the graph
        - Each node has an intensity value and a label (foreground or background)
        - Each node is connected to its 8-neighbors

        - Graph is represented as a dictionary of nodes to (adjacent node, weight) tuples
        - Weights are set with a provided function in the build_graph method
    """
    class Label(enum.Enum):
        FOREGROUND = 0
        BACKGROUND = 1

    class Node:
        """
        A node in the graph representing a image pixel 

        Relative location of the Nodes matter, so we store intensity and location, as well 
        as the label (foreground or background) of the node
        """

        def __init__(self, row: int, col: int, intensity=0, label=None):
            self.row = row
            self.col = col
            self.intensity = self.set_intensity(intensity)      # TODO: Do we need intensity in the node?

            self.label = label

        def __repr__(self) -> str:
            return f"(R: {self.row}, C: {self.col}) -> (I: {self.intensity}, L: {self.label})" 

        def __hash__(self) -> int:
            return hash((self.row, self.col, self.intensity))   

        def __eq__(self, o: object) -> bool:
            if not isinstance(o, ImageGraph.Node):
                return False
            return self.row == o.row and self.col == o.col
        
        def set_intensity(self, intensity):
            """Clip intensity to [0,255] or [0,1.0] if input is int or float respectively

            Args:
                intensity (Real Num): pixel intensity value 
            """
            if type(intensity) == int:
                self.intensity = max(0, min(intensity, 255)) # Clip to [0,255]
            else:
                self.intensity = max(0.0, min(intensity, 1.0)) # Clip to [0,1.0]

    def __init__(self, image: np.ndarray):
        """Init function for imageGraph.

        Args:
            image (WxHx1 numpy array): Numpy array representation of the image 
                - W: Width of the image
                - H: Height of the image
                - 1: Number of channels in the image (1 for greyscale, 3 for RGB), only support greyscale for now
        """
        self.image = image.copy()
        self.height, self.width = image.shape[:2]
        self.N = self.height * self.width

        # Create source and sink nodes
        self.source = self.Node(-1, 0, label=self.Label.FOREGROUND)
        self.sink = self.Node(0, -1, label=self.Label.BACKGROUND)

        # Graph is dictionary of nodes to (adjacent node, weight) tuples
        self.graph = {}  

    def build_graph(self, weight_function: Callable[[np.ndarray, int, int, int, int], float]) -> None:
        def get_neighbors(row, col):
            """
            Returns a list of the 8-neighbors of a pixel
            """
            neighbors = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    if 0 <= row + i < self.height and 0 <= col + j < self.width:
                        neighbors.append((row + i, col + j))
            return neighbors
        
        # Run through all pixels in the image 
        # Create vertice, add intensity, and then add all edges 
        for row in range(self.height):
            for col in range(self.width):
                node = self.Node(row, col, self.image[row, col])
                if node not in self.graph:
                        self.graph[node] = []

                for n_row, n_col in get_neighbors(row, col):
                    neighbor = self.Node(n_row, n_col, self.image[n_row, n_col])
                    self.graph[node].append((neighbor, weight_function(self.image, row, col, n_row, n_col)))

    def add_source_sink(self, regional_weight_function: Callable[[ImageGraph.Node, bool], float]) -> None:
        for node in self.graph:
            source_weight = regional_weight_function(node, True) 
            sink_weight = regional_weight_function(node, False)

            self.graph[node].append((self.source, source_weight))
            self.graph[self.source].append((node, source_weight))
            self.graph[self.sink].append((node, sink_weight))
            self.graph[node].append((self.sink, sink_weight))

    #############################
    # QOL Methods
    #############################
    def get_edges(self, node: ImageGraph.Node) -> list[(ImageGraph.Node, float)]:
        """Get all of the edges for a given node

        Args:
            node (ImageGraph.Node): Node

        Returns:
            list[(ImageGraph.Node, Weight)]: Neighbors and weights of the node
        """
        return self.graph[node]
    
    def get_max_weight(self, node: ImageGraph.Node) -> float:
        """Get the maximum weight of a node

        Args:
            node (ImageGraph.Node): Node

        Returns:
            Weight: Maximum weight of the node
        """
        return max(self.graph[node], key=lambda x: x[1])[1]

    def set_weight(self, node: ImageGraph.Node, neighbor: ImageGraph.Node, weight: float) -> None:
        """Set the weight of an edge

        Args:
            node (ImageGraph.Node): Node
            neighbor (ImageGraph.Node): Neighbor of the node
            weight (float): Weight of the edge
        """
        # Not horrible since we have edge numbers for a node is bounded to 9 
        for idx, (n, w) in enumerate(self.graph[node]):
            if n == neighbor:
                self.graph[node][idx] = (n, weight)
                break 