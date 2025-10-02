"""
Network Topology Management for DFL
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class NetworkTopology(ABC):
    """
    Abstract base class for network topologies
    """

    def __init__(self, num_nodes: int, **kwargs):
        """
        Initialize network topology

        Args:
            num_nodes: Total number of nodes
            **kwargs: Additional topology-specific parameters
        """
        self.num_nodes = num_nodes
        self.adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    @abstractmethod
    def build_topology(self) -> None:
        """Build the network topology"""
        pass

    def get_neighbors(self, node_id: int) -> List[int]:
        """
        Get neighbors for a given node

        Args:
            node_id: Node ID

        Returns:
            List of neighbor node IDs
        """
        if node_id >= self.num_nodes:
            raise ValueError(f"Node ID {node_id} exceeds number of nodes {self.num_nodes}")

        neighbors = []
        for i in range(self.num_nodes):
            if self.adjacency_matrix[node_id, i] == 1:
                neighbors.append(i)

        return neighbors

    def get_all_neighbors(self) -> Dict[int, List[int]]:
        """
        Get neighbors for all nodes

        Returns:
            Dictionary mapping node_id to list of neighbors
        """
        return {node_id: self.get_neighbors(node_id) for node_id in range(self.num_nodes)}

    def get_topology_info(self) -> Dict[str, Any]:
        """
        Get topology information

        Returns:
            Dictionary with topology statistics
        """
        total_edges = np.sum(self.adjacency_matrix) // 2  # Undirected graph
        avg_degree = np.mean(np.sum(self.adjacency_matrix, axis=1))

        return {
            'num_nodes': self.num_nodes,
            'total_edges': total_edges,
            'average_degree': avg_degree,
            'adjacency_matrix': self.adjacency_matrix.tolist()
        }


class RingTopology(NetworkTopology):
    """
    Ring topology where each node connects to its immediate neighbors
    """

    def __init__(self, num_nodes: int, **kwargs):
        super().__init__(num_nodes, **kwargs)
        self.build_topology()

    def build_topology(self) -> None:
        """Build ring topology"""
        for i in range(self.num_nodes):
            # Connect to next node (with wrap-around)
            next_node = (i + 1) % self.num_nodes
            prev_node = (i - 1) % self.num_nodes

            # Bidirectional connections
            self.adjacency_matrix[i, next_node] = 1
            self.adjacency_matrix[i, prev_node] = 1

        logger.info(f"Built ring topology with {self.num_nodes} nodes")


class MeshTopology(NetworkTopology):
    """
    Full mesh topology where every node connects to every other node
    """

    def __init__(self, num_nodes: int, **kwargs):
        super().__init__(num_nodes, **kwargs)
        self.build_topology()

    def build_topology(self) -> None:
        """Build full mesh topology"""
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    self.adjacency_matrix[i, j] = 1

        logger.info(f"Built mesh topology with {self.num_nodes} nodes")


class StarTopology(NetworkTopology):
    """
    Star topology with one central node connected to all others
    """

    def __init__(self, num_nodes: int, center_node: int = 0, **kwargs):
        self.center_node = center_node
        super().__init__(num_nodes, **kwargs)
        self.build_topology()

    def build_topology(self) -> None:
        """Build star topology"""
        for i in range(self.num_nodes):
            if i != self.center_node:
                # Connect to center node (bidirectional)
                self.adjacency_matrix[i, self.center_node] = 1
                self.adjacency_matrix[self.center_node, i] = 1

        logger.info(f"Built star topology with center node {self.center_node}")


class RandomTopology(NetworkTopology):
    """
    Random topology where connections are made with given probability
    """

    def __init__(self, num_nodes: int, connection_prob: float = 0.3,
                 min_degree: int = 1, seed: int = 42, **kwargs):
        self.connection_prob = connection_prob
        self.min_degree = min_degree
        self.seed = seed
        np.random.seed(seed)
        super().__init__(num_nodes, **kwargs)
        self.build_topology()

    def build_topology(self) -> None:
        """Build random topology"""
        # Create random connections
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if np.random.random() < self.connection_prob:
                    self.adjacency_matrix[i, j] = 1
                    self.adjacency_matrix[j, i] = 1

        # Ensure minimum degree for all nodes
        for i in range(self.num_nodes):
            degree = np.sum(self.adjacency_matrix[i])
            if degree < self.min_degree:
                # Find nodes to connect to
                available_nodes = [j for j in range(self.num_nodes)
                                 if i != j and self.adjacency_matrix[i, j] == 0]

                if available_nodes:
                    connections_needed = min(self.min_degree - degree, len(available_nodes))
                    chosen_nodes = np.random.choice(available_nodes, connections_needed, replace=False)

                    for j in chosen_nodes:
                        self.adjacency_matrix[i, j] = 1
                        self.adjacency_matrix[j, i] = 1

        logger.info(f"Built random topology with {self.num_nodes} nodes "
                   f"(p={self.connection_prob}, min_degree={self.min_degree})")


def create_topology(topology_type: str, num_nodes: int, **kwargs) -> NetworkTopology:
    """
    Factory function to create network topology

    Args:
        topology_type: Type of topology ('ring', 'mesh', 'star', 'random')
        num_nodes: Number of nodes
        **kwargs: Additional parameters

    Returns:
        NetworkTopology instance
    """
    topology_type = topology_type.lower()

    if topology_type == 'ring':
        return RingTopology(num_nodes, **kwargs)
    elif topology_type == 'mesh':
        return MeshTopology(num_nodes, **kwargs)
    elif topology_type == 'star':
        return StarTopology(num_nodes, **kwargs)
    elif topology_type == 'random':
        return RandomTopology(num_nodes, **kwargs)
    else:
        raise ValueError(f"Unsupported topology type: {topology_type}")