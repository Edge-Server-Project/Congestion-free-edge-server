"""Creation and connectivity management for an edge-node network."""

from __future__ import annotations

import random
from collections.abc import Iterator

from config import NetworkConfig
from models import EdgeNode


class Network:
    """A deterministic, undirected network of configured edge nodes."""

    def __init__(self, config: NetworkConfig) -> None:
        """Create the configured edge nodes and build their connectivity."""
        self.config = config
        self._rng = random.Random(config.seed)
        self._nodes: dict[str, EdgeNode] = {}
        self.create_edge_nodes()
        self.build_connectivity()

    @property
    def nodes(self) -> tuple[EdgeNode, ...]:
        """Return all edge nodes in creation order."""
        return tuple(self._nodes.values())

    def create_edge_nodes(self) -> None:
        """Create the configured edge nodes using the configured random seed."""
        self._rng = random.Random(self.config.seed)
        self._nodes.clear()
        for index in range(1, self.config.num_edge_nodes + 1):
            node = EdgeNode(
                node_id=f"E{index}",
                x=self._rng.uniform(0.0, self.config.area_width),
                y=self._rng.uniform(0.0, self.config.area_height),
                processing_capacity=self.config.processing_capacity,
                queue_capacity=self.config.queue_capacity,
                initial_energy=self.config.initial_energy,
            )
            self._add_node(node)

    def build_connectivity(self) -> None:
        """Build a symmetric range-based topology without self-links."""
        for node in self.nodes:
            node.clear_neighbours()

        nodes = self.nodes
        for index, source in enumerate(nodes):
            for destination in nodes[index + 1 :]:
                if source.distance_to(destination) <= self.config.communication_range:
                    source.add_neighbour(destination.node_id)
                    destination.add_neighbour(source.node_id)

    def get_node(self, node_id: str) -> EdgeNode:
        """Return an edge node by ID, or raise a clear error when it is unknown."""
        try:
            return self._nodes[node_id]
        except KeyError as error:
            raise KeyError(f"unknown edge node: {node_id}") from error

    def get_neighbours(self, node_id: str) -> tuple[EdgeNode, ...]:
        """Return the edge nodes connected to the requested edge node."""
        node = self.get_node(node_id)
        return tuple(self.get_node(neighbour_id) for neighbour_id in node.neighbours)

    def iter_edges(self) -> Iterator[tuple[str, str]]:
        """Yield each undirected network link exactly once."""
        for node in self.nodes:
            for neighbour_id in node.neighbours:
                if node.node_id < neighbour_id:
                    yield (node.node_id, neighbour_id)

    def _add_node(self, node: EdgeNode) -> None:
        """Store a node after checking its identity and configured bounds."""
        if node.node_id in self._nodes:
            raise ValueError(f"duplicate node ID: {node.node_id}")
        if not 0.0 <= node.x <= self.config.area_width:
            raise ValueError(f"node {node.node_id} has an x position outside the network area")
        if not 0.0 <= node.y <= self.config.area_height:
            raise ValueError(f"node {node.node_id} has a y position outside the network area")
        self._nodes[node.node_id] = node
