"""
P2P Communication Protocol for DFL
"""

from typing import Dict, List, Any, Optional, Tuple
import torch
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CommunicationProtocol(ABC):
    """
    Abstract base class for P2P communication protocols
    """

    def __init__(self, node_id: int, neighbors: List[int]):
        """
        Initialize communication protocol

        Args:
            node_id: ID of the current node
            neighbors: List of neighbor node IDs
        """
        self.node_id = node_id
        self.neighbors = neighbors
        self.message_buffer = {}

    @abstractmethod
    def send_model(self, model_params: Dict[str, torch.Tensor],
                   target_node: int) -> bool:
        """Send model parameters to target node"""
        pass

    @abstractmethod
    def receive_models(self) -> List[Tuple[int, Dict[str, torch.Tensor]]]:
        """Receive model parameters from neighbors"""
        pass

    @abstractmethod
    def broadcast_model(self, model_params: Dict[str, torch.Tensor]) -> bool:
        """Broadcast model parameters to all neighbors"""
        pass


class SimulatedP2PProtocol(CommunicationProtocol):
    """
    Simulated P2P communication protocol for experiments
    Uses shared memory for communication simulation
    """

    # Shared communication buffer for all nodes
    _global_buffer: Dict[int, Dict[str, Any]] = {}

    def __init__(self, node_id: int, neighbors: List[int]):
        super().__init__(node_id, neighbors)

        # Initialize node's mailbox in global buffer
        if node_id not in self._global_buffer:
            self._global_buffer[node_id] = {
                'inbox': [],
                'outbox': [],
                'metadata': {'round': 0}
            }

        logger.info(f"Node {node_id} communication initialized with neighbors: {neighbors}")

    def send_model(self, model_params: Dict[str, torch.Tensor],
                   target_node: int) -> bool:
        """
        Send model parameters to target node

        Args:
            model_params: Model parameters to send
            target_node: Target node ID

        Returns:
            Success status
        """
        try:
            if target_node not in self._global_buffer:
                self._global_buffer[target_node] = {
                    'inbox': [],
                    'outbox': [],
                    'metadata': {'round': 0}
                }

            # Create message
            message = {
                'sender': self.node_id,
                'model_params': {k: v.clone() for k, v in model_params.items()},
                'timestamp': self._get_timestamp()
            }

            # Add to target's inbox
            self._global_buffer[target_node]['inbox'].append(message)

            logger.debug(f"Node {self.node_id} sent model to node {target_node}")
            return True

        except Exception as e:
            logger.error(f"Failed to send model from {self.node_id} to {target_node}: {e}")
            return False

    def receive_models(self) -> List[Tuple[int, Dict[str, torch.Tensor]]]:
        """
        Receive model parameters from neighbors

        Returns:
            List of (sender_id, model_params) tuples
        """
        received_models = []

        if self.node_id in self._global_buffer:
            inbox = self._global_buffer[self.node_id]['inbox']

            while inbox:
                message = inbox.pop(0)
                sender_id = message['sender']
                model_params = message['model_params']
                received_models.append((sender_id, model_params))

        logger.debug(f"Node {self.node_id} received {len(received_models)} models")
        return received_models

    def broadcast_model(self, model_params: Dict[str, torch.Tensor]) -> bool:
        """
        Broadcast model parameters to all neighbors

        Args:
            model_params: Model parameters to broadcast

        Returns:
            Success status
        """
        success_count = 0

        for neighbor_id in self.neighbors:
            if self.send_model(model_params, neighbor_id):
                success_count += 1

        success = success_count == len(self.neighbors)
        logger.info(f"Node {self.node_id} broadcast to {success_count}/{len(self.neighbors)} neighbors")

        return success

    def _get_timestamp(self) -> int:
        """Get current timestamp (simplified)"""
        import time
        return int(time.time() * 1000)

    def clear_buffers(self) -> None:
        """Clear communication buffers"""
        if self.node_id in self._global_buffer:
            self._global_buffer[self.node_id]['inbox'].clear()
            self._global_buffer[self.node_id]['outbox'].clear()

    @classmethod
    def reset_global_buffer(cls) -> None:
        """Reset global communication buffer"""
        cls._global_buffer.clear()

    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status for debugging"""
        if self.node_id in self._global_buffer:
            buffer = self._global_buffer[self.node_id]
            return {
                'inbox_size': len(buffer['inbox']),
                'outbox_size': len(buffer['outbox']),
                'metadata': buffer['metadata']
            }
        return {'inbox_size': 0, 'outbox_size': 0, 'metadata': {}}