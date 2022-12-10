#############################
# Various Graph Cut Algorithms which operate on the ImageGraph 
# representation of an image. Seeding of any sort is represented in the 
# ImageGraph object. These algorithms should further be able to label 
# the non-seeded pixels in the ImageGraph object they are given. 
#
# Matthew Hokinson, 12/7/22
#############################

from __future__ import annotations
from collections import deque

from imageGraph import ImageGraph 

def pushRelabelCut(imageGraph: ImageGraph) -> None:
    """
    Push-Relabel Graph Cut Algorithm

    The results of the cut are stored in the given ImageGraph object. We run 
    a push-relabel max flow algorithm. The algorithm terminates when there are no remaining 
    paths from the source to the sink.

    Ref: https://cp-algorithms.com/graph/push-relabel.html#implementation

    Args:
        imageGraph (ImageGraph): ImageGraph representation of the image
    """
    # Need to track current height, excess, flow, and capacity of each node (capacity in graph) 
    heights = {node: 0 for node in imageGraph.get_vertices()}
    excess = {node: 0 for node in imageGraph.get_vertices()}
    flow = {node: {neighbor: 0 for (neighbor, _) in imageGraph.get_edges(node)} for node in imageGraph.get_vertices()}
    excessQueue = deque()

    def _push(node: ImageGraph.Node, neighbor: ImageGraph.Node) -> None:
        """
        Push operation for the push-relabel algorithm

        Args:
            node (ImageGraph.Node): Node to push from
            neighbor (ImageGraph.Node): Node to push to
        """
        # Find the amount to push
        pushAmount = min(excess[node], imageGraph.get_edge_weight(node, neighbor) - flow[node][neighbor])

        # Update the flow
        flow[node][neighbor] += pushAmount
        flow[neighbor][node] -= pushAmount # negative flow in the revese direction, should be ok even with undirected graph 

        # Update the excess
        excess[node] -= pushAmount
        excess[neighbor] += pushAmount

        # Add neighbor to the queue if it has excess 
        # If the excess is the push amount, then we know it wasn't in the queue before
        if excess[neighbor] == pushAmount:
            excessQueue.appendLeft(neighbor)
    
    def _relabel(node: ImageGraph.Node) -> None:
        """
        Relabel operation for the push-relabel algorithm. Set the height 
        of node to be the minimum height of its neighbors + 1

        Args:
            node (ImageGraph.Node): Node to relabel
        """
        # Find the minimum height of the neighbors where there is some remaining capacity
        neighborsWithCapacity = [neighbor for neighbor in imageGraph.get_neighbors(node) if flow[node][neighbor] < imageGraph.get_edge_weight(node, neighbor)]

        # If there are no neighbors with capacity, then we can't relabel
        if len(neighborsWithCapacity) == 0:
            return

        # Update the height
        minNeighborHeight = min(heights[neighbor] for neighbor in neighborsWithCapacity)
        heights[node] = minNeighborHeight + 1

    def _discharge(node: ImageGraph.Node) -> None:
        """
        Discharge operation for the push-relabel algorithm. Pushes as much 
        as possible from the node, then relabels if necessary

        Args:
            node (ImageGraph.Node): Node to discharge
        """
        while excess[node] > 0:
            # Find the neighbors with capacity and lower height 
            neighborsWithCapacity = [neighbor for neighbor in imageGraph.get_neighbors(node) 
                if flow[node][neighbor] < imageGraph.get_edge_weight(node, neighbor) and heights[node] > heights[neighbor]]

            # If there are no neighbors with capacity, then we need to relabel
            if len(neighborsWithCapacity) == 0:
                _relabel(node)
                continue

            # Push to the neightbors while we have excess 
            for neighbor in neighborsWithCapacity:
                if excess[node] <= 0:
                    break
                _push(node, neighbor)

    # Initialize the height and preflow for the source
    if imageGraph.source is None or imageGraph.sink is None:
        raise ValueError("Source or Sink is not set!")

    heights[imageGraph.source] = imageGraph.N
    sumSourceWeights = sum(imageGraph.get_edges(imageGraph.source).values())
    excess[imageGraph.source] = sumSourceWeights
    for neighbor in imageGraph.get_neighbors(imageGraph.source):
        _push(imageGraph.source, neighbor)

    while len(excessQueue) > 0:
        node = excessQueue.pop(0)

        if node == imageGraph.source or node == imageGraph.sink:
            continue
        _discharge(node)

    # Set the labels of the nodes in the imageGraph based on nodes which 
    # node still have capacity to the source (foreground) or sink (background)
    # Trivially, nodes can only have remaining capacity in one or the other, otherwise there 
    # would still be a path from the source to the sink
    for node in imageGraph.get_vertices():
        if node == imageGraph.source or node == imageGraph.sink:
            continue

        # If there is still capacity from source to node, then it is foreground
        # We will have subtracted flow from source to node if we pushed back
        if flow[imageGraph.source][node] < imageGraph.get_edge_weight(node, imageGraph.source):
            imageGraph.set_label(node, ImageGraph.Label.Foreground)
        else:
            imageGraph.set_label(node, ImageGraph.Label.Background)