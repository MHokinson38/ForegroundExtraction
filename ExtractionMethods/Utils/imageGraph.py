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

    Meant to be used to store both the orinal image and the graph representation 
        => Do not store the Image in the specific cut implementation class  

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

        Relative location of the Nodes matter, so we store location, as well 
        as the label (foreground or background) of the node
        """

        def __init__(self, row: int, col: int, label=None):
            self.row = row
            self.col = col

            self.label = label

        def __repr__(self) -> str:
            return f"(R: {self.row}, C: {self.col}) -> (Label: {self.label})" 

        def __hash__(self) -> int:
            return hash((self.row, self.col))   

        def __eq__(self, o: object) -> bool:
            if not isinstance(o, ImageGraph.Node):
                return False
            return self.row == o.row and self.col == o.col

    # Type Aliases 
    graph_t = dict[Node, list[tuple[Node, float]]]

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

        # Create source and sink nodes
        self.source = self.__get_or_create_node(-1, 0, label=self.Label.FOREGROUND)
        self.sink = self.__get_or_create_node(0, -1, label=self.Label.BACKGROUND)

        # Graph is dictionary of nodes to a list of (adjacent node, weight) tuples
        self.graph = {}  
        self.__node_lookup = {}      # For mapping index to node, this is SSOT for nodes 
        self.__seeded_pixels = set() # Set of seeded pixels

    def build_graph(self, weight_function: Callable[[ImageGraph.Node, ImageGraph.Node, np.ndarray], int]) -> None:
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
                node = self.__get_or_create_node(row, col)

                for n_row, n_col in get_neighbors(row, col):
                    neighbor = self.__get_or_create_node(n_row, n_col)
                    self.graph[node].append((neighbor, weight_function(node, neighbor, self.image)))

    def add_weighted_source_sink(self, regional_weight_function: Callable[[ImageGraph.Node, bool, np.ndarray], int]) -> None:
        for node in self.graph:
            source_weight = regional_weight_function(node, True, self.image) 
            sink_weight = regional_weight_function(node, False, self.image)

            self.graph[node].append((self.source, source_weight))
            self.graph[self.source].append((node, source_weight))
            self.graph[self.sink].append((node, sink_weight))
            self.graph[node].append((self.sink, sink_weight))

    def set_seeds(self, foregroundSeeds: list[(int, int)], backgroundSeeds: list[(int, int)]) -> None:
        for row, col in foregroundSeeds:
            node = self.__get_or_create_node(row, col)
            node.label = self.Label.FOREGROUND
            self.__seeded_pixels.add(node)

        for row, col in backgroundSeeds:
            node = self.__get_or_create_node(row, col)
            node.label = self.Label.BACKGROUND
            self.__seeded_pixels.add(node)

    def get_seeded_pixels(self) -> graph_t:
        return {node: self.graph[node] for node in self.__seeded_pixels}

    #############################
    # Private methods 
    #############################
    def __get_node(self, row: int, col: int) -> ImageGraph.Node:
        if (row, col) not in self.__node_lookup:
            self.__node_lookup[(row, col)] = self.Node(row, col)
        return self.__node_lookup[(row, col)]

    def __get_or_create_node(self, row: int, col: int, label=None) -> ImageGraph.Node:
        if (row, col) not in self.__node_lookup:
            self.__node_lookup[(row, col)] = self.Node(row, col)
            self.graph[self.__node_lookup[(row, col)]] = []
        return self.__node_lookup[(row, col)]

    def __node_present(self, row: int, col: int) -> bool:
        return (row, col) in self.__node_lookup

    #############################
    # Image Processing 
    #############################
    def get_foreground(self) -> np.ndarray:
        foreground = np.zeros_like(self.image)
        for row in range(self.height):
            for col in range(self.width):
                node = self.__get_node(row, col)
                if node.label == self.Label.FOREGROUND:
                    foreground[row, col] = self.image[row, col]

        return foreground

    def get_foreground_mask(self) -> np.ndarray:
        mask = np.zeros_like(self.image)
        for row in range(self.height):
            for col in range(self.width):
                node = self.__get_node(row, col)
                if node.label == self.Label.FOREGROUND:
                    mask[row, col] = 255

        return mask

    #############################
    # QOL Methods (For random outside use)
    #############################
    def get_edges(self, node: ImageGraph.Node) -> list[(ImageGraph.Node, float)]:
        """Get all of the edges for a given node

        Args:
            node (ImageGraph.Node): Node

        Returns:
            list[(ImageGraph.Node, Weight)]: Neighbors and weights of the node
        """
        if not self.__node_present(node.row, node.col):
            raise ValueError("Node not in graph")

        return self.graph[node]
    
    def get_max_weight(self, node: ImageGraph.Node) -> float:
        """Get the maximum weight of a node

        Args:
            node (ImageGraph.Node): Node

        Returns:
            Weight: Maximum weight of the node
        """
        if not self.__node_present(node.row, node.col):
            raise ValueError("Node not in graph")

        return max(self.graph[node], key=lambda x: x[1])[1]

    def set_weight(self, node: ImageGraph.Node, neighbor: ImageGraph.Node, weight: float) -> None:
        """Set the weight of an edge

        Args:
            node (ImageGraph.Node): Node
            neighbor (ImageGraph.Node): Neighbor of the node
            weight (float): Weight of the edge
        """
        if not self.__node_present(node.row, node.col) or not self.__node_present(neighbor.row, neighbor.col):
            raise ValueError("Node or neighbor not in graph")

        # Not horrible since we have edge numbers for a node is bounded to 9 
        for idx, (n, _) in enumerate(self.graph[node]):
            if n == neighbor:
                self.graph[node][idx] = (n, weight)
                break 