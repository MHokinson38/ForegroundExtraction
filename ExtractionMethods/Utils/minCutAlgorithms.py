from __future__ import annotations
#############################
# Various Graph Cut Algorithms which operate on the ImageGraph 
# representation of an image. Seeding of any sort is represented in the 
# ImageGraph object. These algorithms should further be able to label 
# the non-seeded pixels in the ImageGraph object they are given. 
#
# Matthew Hokinson, 12/7/22
#############################

from collections import deque
from typing import Tuple
import maxflow  
import math 

from .imageGraph import ImageGraph 

def biggerBetterFasterStronger(imageGraph: ImageGraph) -> None:
    """
    Max Flow Graph Cut Algorithm

    The results of the cut are stored in the given ImageGraph object. We run 
    a max flow algorithm. The algorithm terminates when there are no remaining 
    paths from the source to the sink.

    Args:
        imageGraph (ImageGraph): ImageGraph representation of the image
    """
    # Create a maxflow graph
    graph = maxflow.Graph[int](imageGraph.N, 3 * imageGraph.N)
    # Add the nodes
    nodes = {node: graph.add_nodes(1) for node in imageGraph.get_vertices()}

    # Add the edges
    for node in imageGraph.get_vertices():
        for neighbor in imageGraph.get_edges(node).keys():
            # Add the edge
            weight = imageGraph.get_edge_weight(node, neighbor)
            graph.add_edge(nodes[node], nodes[neighbor], weight, weight)

    # Add the source and sink edge
    for node in imageGraph.get_vertices():
        if node == imageGraph.source or node == imageGraph.sink:
            continue
            
        # Add the edge
        graph.add_tedge(nodes[node], imageGraph.get_edge_weight(node, imageGraph.source), 
                        imageGraph.get_edge_weight(node, imageGraph.sink))

    # Run the max flow algorithm
    graph.maxflow()

    # Label the nodes
    for node in imageGraph.get_vertices():
        # print(f"Node {node} has segment {graph.get_segment(nodes[node])}")
        if graph.get_segment(nodes[node]) == 0:
            imageGraph.set_label(node, ImageGraph.Label.FOREGROUND)
        else:
            imageGraph.set_label(node, ImageGraph.Label.BACKGROUND)

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
    flow = {node: {neighbor: 0 for neighbor in imageGraph.get_edges(node).keys()} for node in imageGraph.get_vertices()}
    excessQueue = deque()

    iterations = 0

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
            excessQueue.appendleft(neighbor)
    
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
        iterations += 1
        if iterations % 100000 == 0:
            print(f"Iterations: {iterations/ 100000} x 10^5")
        node = excessQueue.pop()

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
            imageGraph.set_label(node, ImageGraph.Label.FOREGROUND)
        else:
            imageGraph.set_label(node, ImageGraph.Label.BACKGROUND)

def boykov_kolmogorov(imageGraph: ImageGraph) -> None: 
    """
    Boykov-Kolmogorov Graph Cut Algorithm

    Steps: 
        1. Construct two search trees, S and T, originating from the source and sink, respectively 
        2. Grow each tree through the graph on all edges that are not saturated
            2.1 Growth stage ends an active node (leaf) reaches another active node (leaf) in the other tree
        3. Augment phase 
            3.1 Augment path found in the growth stage (two trees, move up to parents until S and T are met)
            3.2 We want to fully saturate the path, with the min capacity of the whole path 
            3.3 This changes the capacities and removes the saturated edges from the tree, which can result 
                in orphans of existing nodes 
        4. Adoption phase 
            4.1 Want to find a new parent for orphan o in the same tree as o, if there is none then we 
                make o a free node 
            4.2 If o becomes free, we declare all o's children as orphans and repeat, until no orphans remain.
        5. Repeat steps 2-4 until no more augmenting paths can be found (i.e. no more active nodes) 
    """
    # State constants (Maybe) 
    FREE = 0
    S_TREE = 1
    T_TREE = 2

    # Initialize the search trees, active nodes, and orphans 
    S = set(imageGraph.source) # All nodes in S
    T = set(imageGraph.sink) # All nodes in T (for both, structure is stored in parents) 
    activeNodes = set([imageGraph.source, imageGraph.sink]) # Active nodes are leaves of the trees
    orphans = set() # Orphans are nodes that have been removed from the tree, but have children

    # Store the parents, but also the augmenting path from node to the terminal node ?
    parents = {imageGraph.source: None, imageGraph.sink: None} # Parents of nodes in the tree
    
    flow = {node: {neighbor: 0 for neighbor in imageGraph.get_neighbors(node)} for node in imageGraph.get_vertices()} # Flow in the graph

    def _grow() -> list[ImageGraph.Node]:
        """
        Grow the search trees from the active nodes. 

        Returns:
            list[ImageGraph.Node]: Augmenting path found during growth, if any
        """
        while (len(activeNodes) > 0):
            # Get the next active node 
            node = activeNodes.pop()

            # Get the neighbors of the node that are not saturated
            neighbors = [neighbor for neighbor in imageGraph.get_neighbors(node) 
                if flow[node][neighbor] < imageGraph.get_edge_weight(node, neighbor)]

            # Add the neighbors to the tree and mark them as active 
            for neighbor in neighbors:
                # If the neighbor is already in the tree, then we have a cycle 
                if neighbor not in S and neighbor not in T:
                    # Return the augmenting path 
                    parents[neighbor] = node
                    if node in S:
                        S.add(neighbor)
                    else:
                        T.add(neighbor)
                else:
                    # Neighbor is in a tree, so we return the augmenting path P, also return P to the active node list 
                    activeNodes.add(node)
                    return _augmenting_path(node, neighbor)

        return None

    def _augmenting_path(node1: ImageGraph.Node, node2: ImageGraph.Node) -> Tuple(list[ImageGraph.Node], int):
        """
        Find the augmenting path between two nodes in the same tree

        Args:
            node1 (ImageGraph.Node): First node
            node2 (ImageGraph.Node): Second node

        Returns:
            list[ImageGraph.Node]: Augmenting path between the two nodes
        """
        def _find_path_to_root(node: ImageGraph.Node) -> Tuple(list[ImageGraph.Node], int):
            """
            Find the path from a node to the root of the tree

            Args:
                node (ImageGraph.Node): Node to find path from

            Returns:
                list[ImageGraph.Node]: Path from node to root
            """
            path = []
            min_weight = math.inf
            while node is not None:
                path.append(node)
                min_weight = min(min_weight, imageGraph.get_edge_weight(parents[node], node))
                node = parents[node]
            return path, min_weight

        path1, min_weight1 = _find_path_to_root(node1)
        path2, min_weight2 = _find_path_to_root(node2)

        min_weight = min(min_weight1, min_weight2, imageGraph.get_edge_weight(node1, node2))
 
        if node1 in S:
            return path1[::-1] + path2, min_weight
        else:
            return path2[::-1] + path1, min_weight

    def _augment(path: list[ImageGraph.Node], min_weight: int) -> None:
        """
        Augment the path by the min_weight

        Args:
            path (list[ImageGraph.Node]): Path to augment
            min_weight (int): Amount to augment by
        """
        for i in range(len(path) - 1):
            flow[path[i]][path[i + 1]] += min_weight
            flow[path[i + 1]][path[i]] -= min_weight

            # check if the edge is saturated
            if flow[path[i]][path[i + 1]] == imageGraph.get_edge_weight(path[i], path[i + 1]):
                # if both nodes in S, then remove edge from S and make the second an orphan 
                if path[i] in S and path[i + 1] in S:
                    parents[path[i + 1]] = None
                    orphans.add(path[i + 1])
                elif path[i] in T and path[i + 1] in T:
                    parents[path[i]] = None
                    orphans.add(path[i])

    def _adopt() -> None:
        """
        Adopt orphans into the tree
        """
        while len(orphans) > 0:
            orphan = orphans.pop()

            # Get neighbors with remaining capacity 
            neighbors = [neighbor for neighbor in imageGraph.get_neighbors(orphan) 
                if flow[orphan][neighbor] < imageGraph.get_edge_weight(orphan, neighbor)]

            # Find a parent which belongs to a tree and has remaining capacity 
            for neighbor in neighbors:
                if neighbor in S or neighbor in T: # TODO: Need to verify that neighbor is connected back to the source/sink 
                    # Adopt the orphan 
                    parents[orphan] = neighbor
                    break

            # If we couldn't find a parent, then the orphan becomes an active node again 
            if parents[orphan] is None:
                activeNodes.appendleft(orphan)

                # Mark other neighbors from the same tree as orphans if need be 
                for neighbor in neighbors:
                    if orphan in S and neighbor in S: 
                        activeNodes.appendleft(neighbor)
                    elif orphan in T and neighbor in T:
                        activeNodes.appendleft(neighbor)
                    
                    if parents[neighbor] == orphan:
                        parents[neighbor] = None
                        orphans.add(neighbor)

                # remove orphan from the tree and the active node set, P is now a free node again 
                if orphan in S:
                    S.remove(orphan)
                elif orphan in T:
                    T.remove(orphan)
                if orphan in activeNodes:
                    activeNodes.remove(orphan)
    # Not entirely finished 
    raise NotImplementedError