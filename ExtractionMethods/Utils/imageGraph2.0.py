from __future__ import annotations
#############################
# Data Structure for storing images as a weighted graph
# for use in graph cut algorithms, but faster 
#
# Matthew Hokinson, 12/13/22
#############################

import numpy as np
import scipy as sp

from collections.abc import Callable
import enum


class ImageGraph:
    """
    Data structure for storing images as a weighted graph for use in graph cut algorithms

    The previous ImageGraph representation was slow, we we are going full numpy and 
    are going to optimize the shit out of this thing (hopefully)
    
    Graph Representation: 
        - Singular 2D array, shape (W, H, (8/4), 1) where: -- May be doing this as a sparse scipy matrix 
            - W: Width of the image
            - H: Height of the image
            - 8/4: Number of edges for each pixel, pending graph connections 
            - 1: Weight of the edge
        - We will also include the terminal edges in another array, which is 
            shape (W, H, 2, 1) where:
            - 2: Number of terminal edges (0 for S, 1 for T) 
            - 1: Weight of the edge
        - Building the graph is going to be the same, while the remaining operations will 
            be different. 

    Ideally, this will replace the existing implementation, and to avoid rewriting 
    all of the code, I suppose we can fit this over the existing interface defined by 
    the ImageGraph class. 

    This means we'll still include the Node subclass (for labels as well), but will 
    be using numpy operations and arrays rather than dictionaries for all lookups and 
    operations, hopefully improving speed. If this still fails to speed things up, 
    it's probably time to use another graph algo libary implemented in C. 
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

        self.use8Neighbors = True

        # Might get away without these, but maybe not. Will 
        # have to change edge weight lookups if we keep them 
        self.source = None
        self.sink = None

        # Graph data structure 
        # self.__graph = np.zeros((self.width, self.height, 8 if self.use8Neighbors else 4), dtype=int)
        self.__graph = sp.sparse.csr_array((self.N, self.N), dtype=int)
        self.__terminalEdges = np.zeros((self.width, self.height, 2), dtype=int)
        self.__nodeLookup = {} # For mapping index to nodes, might still need this, though I may remove node entirely
        self.__seededNodes = np.zeros((self.width, self.height), dtype=bool) # For keeping track of seeded nodes

    def build_graph(self, weight_function: Callable[[ImageGraph.Node, ImageGraph.Node, np.ndarray], int]) -> None:
        def get_neighbors(row, col):
            """
            Returns a list of the 8-neighbors of a pixel
            """
            # 8-neighbor offsets of the pixel
            offsets8 = [(-1,0), (0,-1), (0,1), (1,0), (-1,-1), (-1,1), (1,-1), (1,1)]
            offsets4 = [(-1,0), (0,-1), (0,1), (1,0)]
            neighbors = []
            for offset in offsets8 if self.use8Neighbors else offsets4:
                n_row, n_col = row + offset[0], col + offset[1]
                if 0 <= n_row < self.height and 0 <= n_col < self.width:
                    neighbors.append((n_row, n_col))
            return neighbors
        
        # Run through all pixels in the image 
        # Create vertice, add intensity, and then add all edges 
        for row in range(self.height):
            for col in range(self.width):
                node = self.__get_or_create_node(row, col) # Should still be creating nodes, but not using them for graph

                for n_row, n_col in get_neighbors(row, col):
                    neighbor = self.__get_or_create_node(n_row, n_col)
                    if not self.__edge_exists(node, neighbor):
                        self.__create_edge(node, neighbor, weight_function(node, neighbor, self.image))

    def add_weighted_source_sink(self, regional_weight_function: Callable[[ImageGraph.Node, bool, np.ndarray], int]) -> None:
        # Verify the source and sink haven't been added yet 
        if self.source is not None or self.sink is not None:
            raise Exception("Source and sink have already been added to the graph")
        
        # Create source and sink nodes (Keep for now) 
        self.source = self.__get_or_create_node(-1, 0, label=self.Label.FOREGROUND)
        self.sink = self.__get_or_create_node(0, -1, label=self.Label.BACKGROUND)

        for row in range(self.height):
            for col in range(self.width):
                self.__terminalEdges[row, col, 0] = regional_weight_function(
                    self.__get_node(row, col), True, self.image)
                self.__terminalEdges[row, col, 1] = regional_weight_function(
                    self.__get_node(row, col), False, self.image)

    def set_seeds(self, foregroundSeeds: list[(int, int)], backgroundSeeds: list[(int, int)]) -> None:
        for row, col in foregroundSeeds:
            self.set_seed(row, col, True)

        for row, col in backgroundSeeds:
            self.set_seed(row, col, False)

    def set_seed(self, row: int, col: int, isForeground: bool) -> None:
        # Validate the pixel location
        if not self.__is_valid_pixel(row, col):
            raise Exception(f"Invalid pixel location ({row}, {col})")

        node = self.__get_or_create_node(row, col)
        node.label = self.Label.FOREGROUND if isForeground else self.Label.BACKGROUND
        self.__seeded_pixels.add(node)

    def set_label(self, node: ImageGraph.Node, label: Label) -> None:
        # Verify that this node is not a seed 
        if not node in self.__seeded_pixels:
            node.label = label

    #############################
    # Graph Stucture Modifications (Private)
    # Since the graph needs to represeent the image, 
    # we cannot modify the structure outside of the class 
    #############################
    def __linear_index(self, row: int, col: int) -> int:
        return row * self.width + col

    def linear_index(self, node: ImageGraph.Node) -> int:
        return node.row * self.width + node.col
    
    def __get_node(self, row: int, col: int) -> ImageGraph.Node:
        if (row, col) not in self.__node_lookup:
            return None
        return self.__node_lookup[(row, col)]

    def __get_or_create_node(self, row: int, col: int, label=None) -> ImageGraph.Node:
        # Verify this is a valid pixel location 
        if not self.__is_valid_pixel(row, col) and label is None: # hacky label check 
            raise ValueError(f"Invalid pixel location ({row}, {col})")

        if (row, col) not in self.__node_lookup:
            self.__node_lookup[(row, col)] = self.Node(row, col)
        return self.__node_lookup[(row, col)]

    def __create_edge(self, node: ImageGraph.Node, neighbor: ImageGraph.Node, weight: int) -> None:
        if not self.__node_present(node):
            raise ValueError(f"Node {node} not present in graph")
        if not self.__node_present(neighbor):
            raise ValueError(f"Node {neighbor} not present in graph")

        self.__graph[self.linear_index(node), self.linear_index(neighbor)] = weight
        self.__graph[self.linear_index(neighbor), self.linear_index(node)] = weight

    def __edge_exists(self, row: int, col: int, nRow: int, nCol: int) -> bool:
        if not self.__is_valid_pixel(row, col):
            raise ValueError(f"Node {row},{col} not present in graph")
        if not self.__is_valid_pixel(nRow, nCol):
            raise ValueError(f"Node {nRow},{nCol} not present in graph")

        return self.__graph[self.__linear_index(row, col), self.__linear_index(nRow, nCol)] != 0

    #############################
    # Validation Checks (Private)
    #############################
    def __is_valid_pixel(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

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
    # Query Methods (For QOL)
    #############################
    def get_vertices(self) -> None:
        raise NotImplementedError("Deprecated in new graph structure")

    def get_neighbors(self, row: int, col: int) -> int:
        """Get all of the edges for a given node

        Args:
            node (ImageGraph.Node): Node

        Returns:
            list[(ImageGraph.Node, Weight)]: Neighbors and weights of the node
        """
        if not self.__is_valid_pixel(row, col):
            raise ValueError("Node not in graph")

        # Get the non-zero indices of the graph indexed with the row, col
        return np.nonzero(self.__graph[self.__linear_index(row, col), :])[0]

    def get_edge_weight(self,  row: int, col: int, nRow: int, nCol: int) -> int:
        """Get the weight of an edge

        Args:
            node (ImageGraph.Node): Node
            neighbor (ImageGraph.Node): Neighbor

        Returns:
            int: Weight of the edge
        """
        if not self.__edge_exists(row, col, nRow, nCol):
            raise ValueError(f"Edge not in graph, {(row, col)} -> {(nRow, nCol)}")

        return self.__graph[self.__linear_index(row, col), self.__linear_index(nRow, nCol)]

    def get_total_weight(self, node: ImageGraph.Node) -> int:
        """Get the total weight of a node

        Args:
            node (ImageGraph.Node): Node

        Returns:
            int: Total weight of the node
        """
        if not self.__is_valid_pixel(node.row, node.col):
            raise ValueError("Node not in graph")

        return sum(self.__graph[self.__linear_index(node.row, node.col), :])


    def get_seeded_pixels(self) -> list[ImageGraph.Node]:
        """Get the seeded portions of the graph 

        Returns:
            list[ImageGraph.Node]: Seeded pixels
        """
        return list(self.__seeded_pixels)
    
    def get_max_weight(self, row, col) -> int:
        """Get the maximum weight of a node

        Args:
            node (ImageGraph.Node): Node

        Returns:
            int: Maximum weight of the node
        """
        if not self.__is_valid_pixel(row, col):
            raise ValueError("Node not in graph")

        return max(self.__graph[self.__linear_index(row, col), :])
                

    