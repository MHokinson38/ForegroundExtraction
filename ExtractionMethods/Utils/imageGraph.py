from __future__ import annotations
#############################
# Data Structure for storing images as a weighted graph
# for use in graph cut algorithms
#
# Matthew Hokinson, 12/6/22
#############################

import numpy as np

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
    graph_t = dict[Node, dict[Node, int]]

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

        self.source = None
        self.sink = None

        # Graph is dictionary of nodes to a list of (adjacent node, weight) tuples
        self.__graph = {}  
        self.__node_lookup = {}      # For mapping index to nodes 
        self.__seeded_pixels = set() # Set of seeded pixels

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
                node = self.__get_or_create_node(row, col)

                for n_row, n_col in get_neighbors(row, col):
                    neighbor = self.__get_or_create_node(n_row, n_col)
                    if not self.__edge_exists(node, neighbor):
                        self.__create_edge(node, neighbor, weight_function(node, neighbor, self.image))

    def add_weighted_source_sink(self, regional_weight_function: Callable[[ImageGraph.Node, bool, np.ndarray], int]) -> None:
        # Verify the source and sink haven't been added yet 
        if self.source is not None or self.sink is not None:
            raise Exception("Source and sink have already been added to the graph")
        
        # Create source and sink nodes
        self.source = self.__get_or_create_node(-1, 0, label=self.Label.FOREGROUND)
        self.sink = self.__get_or_create_node(0, -1, label=self.Label.BACKGROUND)

        for node in self.__graph:
            # Skip the source and sink (no self edges) 
            if node == self.source or node == self.sink:
                continue
        
            source_weight = regional_weight_function(node, True, self.image) 
            sink_weight = regional_weight_function(node, False, self.image)

            self.__create_edge(node, self.source, source_weight)
            self.__create_edge(node, self.sink, sink_weight)

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
            self.__graph[self.__node_lookup[(row, col)]] = {}
        return self.__node_lookup[(row, col)]

    def __node_present(self, n: ImageGraph.Node) -> bool:
        return (n.row, n.col) in self.__node_lookup

    def __create_edge(self, node: ImageGraph.Node, neighbor: ImageGraph.Node, weight: int) -> None:
        if not self.__node_present(node):
            raise ValueError(f"Node {node} not present in graph")
        if not self.__node_present(neighbor):
            raise ValueError(f"Node {neighbor} not present in graph")

        self.__graph[node][neighbor] = weight
        self.__graph[neighbor][node] = weight # Undirected graph

    def __edge_exists(self, node: ImageGraph.Node, neighbor: ImageGraph.Node) -> bool:
        if not self.__node_present(node):
            raise ValueError(f"Node {node} not present in graph")
        if not self.__node_present(neighbor):
            raise ValueError(f"Node {neighbor} not present in graph")

        return neighbor in self.__graph[node] # So long as we only use create_edge, we only need to check one direction

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
    def get_vertices(self) -> list[ImageGraph.Node]:
        """Get all of the vertices in the graph

        Returns:
            list[ImageGraph.Node]: List of all nodes in the graph
        """
        return list(self.__graph.keys())

    def get_edges(self, node: ImageGraph.Node) -> dict[Node, int]:
        """Get all of the edges for a given node

        Args:
            node (ImageGraph.Node): Node

        Returns:
            list[(ImageGraph.Node, Weight)]: Neighbors and weights of the node
        """
        if not self.__node_present(node):
            raise ValueError("Node not in graph")

        return self.__graph[node]

    def get_edge_weight(self, node: ImageGraph.Node, neighbor: ImageGraph.Node) -> int:
        """Get the weight of an edge

        Args:
            node (ImageGraph.Node): Node
            neighbor (ImageGraph.Node): Neighbor

        Returns:
            int: Weight of the edge
        """
        if not self.__edge_exists(node, neighbor):
            raise ValueError(f"Edge not in graph, {node} -> {neighbor}")

        return self.__graph[node][neighbor]

    def get_total_weight(self, node: ImageGraph.Node) -> int:
        """Get the total weight of a node

        Args:
            node (ImageGraph.Node): Node

        Returns:
            int: Total weight of the node
        """
        if not self.__node_present(node):
            raise ValueError("Node not in graph")

        return sum(self.__graph[node].values())

    def get_seeded_pixels(self) -> list[ImageGraph.Node]:
        """Get the seeded portions of the graph 

        Returns:
            list[ImageGraph.Node]: Seeded pixels
        """
        return list(self.__seeded_pixels)

    def get_neighbors(self, node: ImageGraph.Node) -> list[ImageGraph.Node]:
        """Get all of the neighbors of a node

        Args:
            node (ImageGraph.Node): Node

        Returns:
            list[ImageGraph.Node]: Neighbors of the node
        """
        if not self.__node_present(node):
            raise ValueError("Node not in graph")

        return list(self.__graph[node].keys())
    
    def get_max_weight(self, node: ImageGraph.Node) -> float:
        """Get the maximum weight of a node

        Args:
            node (ImageGraph.Node): Node

        Returns:
            Weight: Maximum weight of the node
        """
        if not self.__node_present(node):
            raise ValueError("Node not in graph")

        return max(self.__graph[node].values())

    def set_weight(self, node: ImageGraph.Node, neighbor: ImageGraph.Node, weight: float) -> None:
        """Set the weight of an edge

        Args:
            node (ImageGraph.Node): Node
            neighbor (ImageGraph.Node): Neighbor of the node
            weight (float): Weight of the edge
        """
        if not self.__edge_exists(node, neighbor):
            raise ValueError("Node or neighbor not in graph")

        self.__graph[node][neighbor] = weight
        self.__graph[neighbor][node] = weight
        