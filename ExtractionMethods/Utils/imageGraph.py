#############################
# Data Structure for storing images as a weighted graph 
# for use in graph cut algorithms
# Currently using Push-Relabel algorithm for flow, so 
# the graph is designed and initalized with this algorithm in mind 
# 
# Matthew Hokinson, 12/6/22
#############################

import numpy as np

import enum 

class ImageGraph:
    """
    Data structure for storing images as a weighted graph for use in graph cut algorithms
    """
    class Label(enum.Enum):
        FOREGROUND = 0
        BACKGROUND = 1

    class Node:
        """
        A node in the graph
        """
        def __init__(self, height, intensity=0, label=None):
            self.height = height
            self.intensity = intensity
            self.label = label


    def __init__(self, image): 
        self.image = image.copy()
        self.height, self.width = image.shape[:2]
        self.N = self.height * self.width

        # Create source and sink nodes 
        self.source = self.Node(self.N, label=self.Label.FOREGROUND)
        self.sink = self.Node(0, label=self.Label.BACKGROUND)

        self.vertices = {}  # dictionary of vertices (nodes)
        self.edges = {}     # dictionary of (unordered pair of nodes) -> weight

