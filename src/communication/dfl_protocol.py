"""
Enhanced DFL Communication Protocol
Provides realistic P2P communication simulation
"""

import torch
import logging
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import time
import threading

logger = logging.getLogger(__name__)


class DFLCommunicationProtocol:
    """
    Enhanced communication protocol for DFL with realistic delays and message passing
    """

    # Global message buffer for all nodes
    _global_message_buffer = {}
    _lock = threading.Lock()

    def __init__(self, node_id: int, neighbors: List[int],
                 latency_ms: float = 10.0, bandwidth_mbps: float = 100.0):
        """
        Initialize communication protocol

        Args:
            node_id: ID of this node
            neighbors: List of neighbor node IDs
            latency_ms: Network latency in milliseconds
            bandwidth_mbps: Network bandwidth in Mbps
        """
        self.node_id = node_id
        self.neighbors = neighbors
        self.latency_ms = latency_ms
        self.bandwidth_mbps = bandwidth_mbps

        # Initialize message buffer for this node
        with self._lock:
            if node_id not in self._global_message_buffer:
                self._global_message_buffer[node_id] = {
                    'inbox': [],
                    'outbox': [],
                    'sent_messages': 0,
                    'received_messages': 0,
                    'total_bytes_sent': 0,
                    'total_bytes_received': 0
                }

        logger.info(f"Node {node_id} communication initialized with neighbors: {neighbors}")

    def send_to_neighbor(self, neighbor_id: int, message: Dict[str, Any]) -> bool:
        """
        Send message to specific neighbor

        Args:
            neighbor_id: Target neighbor ID
            message: Message to send

        Returns:
            Success status
        """
        if neighbor_id not in self.neighbors:
            logger.warning(f"Node {self.node_id}: {neighbor_id} is not a neighbor")
            return False

        try:
            # Calculate message size (approximate)
            message_size = self._calculate_message_size(message)

            # Simulate network delay
            transmission_delay = self._calculate_transmission_delay(message_size)

            # Prepare message with metadata
            full_message = {
                'sender_id': self.node_id,
                'receiver_id': neighbor_id,
                'content': message,
                'timestamp': time.time(),
                'size_bytes': message_size,
                'delay_ms': transmission_delay
            }

            # Send message (add to receiver's inbox)
            with self._lock:
                if neighbor_id not in self._global_message_buffer:
                    self._global_message_buffer[neighbor_id] = {
                        'inbox': [],
                        'outbox': [],
                        'sent_messages': 0,
                        'received_messages': 0,
                        'total_bytes_sent': 0,
                        'total_bytes_received': 0
                    }

                self._global_message_buffer[neighbor_id]['inbox'].append(full_message)

                # Update sender statistics
                self._global_message_buffer[self.node_id]['sent_messages'] += 1
                self._global_message_buffer[self.node_id]['total_bytes_sent'] += message_size

            logger.debug(f"Node {self.node_id} sent message to {neighbor_id} "
                        f"({message_size} bytes, {transmission_delay:.2f}ms delay)")
            return True

        except Exception as e:
            logger.error(f"Failed to send message from {self.node_id} to {neighbor_id}: {e}")
            return False

    def broadcast_to_neighbors(self, message: Dict[str, Any]) -> Dict[int, bool]:
        """
        Broadcast message to all neighbors

        Args:
            message: Message to broadcast

        Returns:
            Dict mapping neighbor_id to success status
        """
        results = {}
        for neighbor_id in self.neighbors:
            results[neighbor_id] = self.send_to_neighbor(neighbor_id, message)

        success_count = sum(results.values())
        logger.info(f"Node {self.node_id} broadcast to {success_count}/{len(self.neighbors)} neighbors")

        return results

    def receive_messages(self) -> List[Dict[str, Any]]:
        """
        Receive all pending messages from inbox

        Returns:
            List of received messages
        """
        messages = []

        with self._lock:
            if self.node_id in self._global_message_buffer:
                inbox = self._global_message_buffer[self.node_id]['inbox']

                while inbox:
                    message = inbox.pop(0)
                    messages.append(message)

                    # Update receiver statistics
                    self._global_message_buffer[self.node_id]['received_messages'] += 1
                    self._global_message_buffer[self.node_id]['total_bytes_received'] += message['size_bytes']

        if messages:
            logger.debug(f"Node {self.node_id} received {len(messages)} messages")

        return messages

    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics for this node"""
        with self._lock:
            if self.node_id in self._global_message_buffer:
                stats = self._global_message_buffer[self.node_id].copy()
                # Remove message queues from stats
                stats.pop('inbox', None)
                stats.pop('outbox', None)
                return stats
            return {}

    def _calculate_message_size(self, message: Dict[str, Any]) -> int:
        """
        Estimate message size in bytes

        Args:
            message: Message to estimate size for

        Returns:
            Estimated size in bytes
        """
        size = 0

        if 'parameters' in message:
            # Calculate tensor sizes
            for param_name, param_tensor in message['parameters'].items():
                if isinstance(param_tensor, torch.Tensor):
                    size += param_tensor.numel() * param_tensor.element_size()
                else:
                    size += 8  # Assume 8 bytes for other data

        # Add overhead for metadata
        size += 1024  # 1KB overhead

        return size

    def _calculate_transmission_delay(self, message_size_bytes: int) -> float:
        """
        Calculate realistic transmission delay

        Args:
            message_size_bytes: Size of message in bytes

        Returns:
            Total delay in milliseconds
        """
        # Base latency
        delay = self.latency_ms

        # Transmission time based on bandwidth
        bandwidth_bytes_per_ms = (self.bandwidth_mbps * 1024 * 1024) / (8 * 1000)
        transmission_time = message_size_bytes / bandwidth_bytes_per_ms

        return delay + transmission_time

    @classmethod
    def reset_global_buffer(cls):
        """Reset global communication buffer"""
        with cls._lock:
            cls._global_message_buffer.clear()

    @classmethod
    def get_global_stats(cls) -> Dict[str, Any]:
        """Get global communication statistics"""
        with cls._lock:
            total_messages = sum(
                node_stats['sent_messages']
                for node_stats in cls._global_message_buffer.values()
            )
            total_bytes = sum(
                node_stats['total_bytes_sent']
                for node_stats in cls._global_message_buffer.values()
            )

            return {
                'total_nodes': len(cls._global_message_buffer),
                'total_messages_sent': total_messages,
                'total_bytes_transmitted': total_bytes,
                'average_message_size': total_bytes / max(total_messages, 1)
            }