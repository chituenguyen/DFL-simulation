"""
Tests for enhanced DFL features
"""

import unittest
import torch
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from communication.dfl_protocol import DFLCommunicationProtocol
from aggregation.fedavg import FedAvgAggregator


class TestDFLCommunicationProtocol(unittest.TestCase):
    """Test enhanced DFL communication protocol"""

    def setUp(self):
        """Set up test fixtures"""
        self.node_id = 0
        self.neighbors = [1, 2]
        self.protocol = DFLCommunicationProtocol(
            node_id=self.node_id,
            neighbors=self.neighbors,
            latency_ms=5.0,
            bandwidth_mbps=50.0
        )

    def tearDown(self):
        """Clean up after tests"""
        DFLCommunicationProtocol.reset_global_buffer()

    def test_protocol_initialization(self):
        """Test protocol initialization"""
        self.assertEqual(self.protocol.node_id, self.node_id)
        self.assertEqual(self.protocol.neighbors, self.neighbors)
        self.assertEqual(self.protocol.latency_ms, 5.0)
        self.assertEqual(self.protocol.bandwidth_mbps, 50.0)

    def test_message_sending(self):
        """Test message sending to neighbor"""
        message = {'test': 'data', 'value': 123}
        neighbor_id = 1

        # Send message
        success = self.protocol.send_to_neighbor(neighbor_id, message)
        self.assertTrue(success)

        # Create receiver protocol to check message
        receiver_protocol = DFLCommunicationProtocol(
            node_id=neighbor_id,
            neighbors=[self.node_id],
            latency_ms=5.0,
            bandwidth_mbps=50.0
        )

        # Receive messages
        messages = receiver_protocol.receive_messages()
        self.assertEqual(len(messages), 1)

        received_message = messages[0]
        self.assertEqual(received_message['sender_id'], self.node_id)
        self.assertEqual(received_message['receiver_id'], neighbor_id)
        self.assertEqual(received_message['content'], message)

    def test_broadcast_to_neighbors(self):
        """Test broadcasting message to all neighbors"""
        message = {'broadcast': 'test'}

        # Broadcast message
        results = self.protocol.broadcast_to_neighbors(message)

        # Check results
        self.assertEqual(len(results), len(self.neighbors))
        for neighbor_id in self.neighbors:
            self.assertIn(neighbor_id, results)
            self.assertTrue(results[neighbor_id])

    def test_send_to_non_neighbor(self):
        """Test sending message to non-neighbor (should fail)"""
        message = {'test': 'data'}
        non_neighbor_id = 99

        success = self.protocol.send_to_neighbor(non_neighbor_id, message)
        self.assertFalse(success)

    def test_message_size_calculation(self):
        """Test message size calculation"""
        # Test with tensor parameters
        message = {
            'parameters': {
                'weight': torch.randn(10, 10),
                'bias': torch.randn(10)
            }
        }

        size = self.protocol._calculate_message_size(message)
        self.assertGreater(size, 0)
        self.assertIsInstance(size, int)

    def test_transmission_delay_calculation(self):
        """Test transmission delay calculation"""
        message_size = 1024  # 1KB

        delay = self.protocol._calculate_transmission_delay(message_size)
        self.assertGreaterEqual(delay, self.protocol.latency_ms)

    def test_communication_stats(self):
        """Test communication statistics tracking"""
        message = {'test': 'data'}

        # Send some messages
        self.protocol.send_to_neighbor(1, message)
        self.protocol.send_to_neighbor(2, message)

        # Get stats
        stats = self.protocol.get_communication_stats()

        self.assertEqual(stats['sent_messages'], 2)
        self.assertGreater(stats['total_bytes_sent'], 0)

    def test_global_stats(self):
        """Test global communication statistics"""
        # Reset for clean test
        DFLCommunicationProtocol.reset_global_buffer()

        # Create multiple protocols and send messages
        protocols = []
        for i in range(3):
            protocol = DFLCommunicationProtocol(
                node_id=i,
                neighbors=[(i + 1) % 3, (i + 2) % 3],
                latency_ms=5.0,
                bandwidth_mbps=50.0
            )
            protocols.append(protocol)

        # Send messages
        message = {'test': 'global_stats'}
        for protocol in protocols:
            protocol.broadcast_to_neighbors(message)

        # Get global stats
        global_stats = DFLCommunicationProtocol.get_global_stats()

        self.assertEqual(global_stats['total_nodes'], 3)
        self.assertGreater(global_stats['total_messages_sent'], 0)
        self.assertGreater(global_stats['total_bytes_transmitted'], 0)


class TestEnhancedFedAvgAggregator(unittest.TestCase):
    """Test enhanced FedAvg aggregator"""

    def setUp(self):
        """Set up test fixtures"""
        # Create sample model parameters
        self.model_params_1 = {
            'weight': torch.randn(5, 5),
            'bias': torch.randn(5)
        }
        self.model_params_2 = {
            'weight': torch.randn(5, 5),
            'bias': torch.randn(5)
        }

        self.model_updates = [
            (0, self.model_params_1),
            (1, self.model_params_2)
        ]

        self.data_sizes = {0: 100, 1: 150}

    def test_aggregator_initialization(self):
        """Test aggregator initialization with enhanced features"""
        aggregator = FedAvgAggregator(
            weighted=True,
            momentum=0.1,
            adaptive_weighting=False
        )

        self.assertTrue(aggregator.weighted)
        self.assertEqual(aggregator.momentum, 0.1)
        self.assertFalse(aggregator.adaptive_weighting)

    def test_aggregation_with_momentum(self):
        """Test aggregation with momentum"""
        aggregator = FedAvgAggregator(weighted=True, momentum=0.2)

        # First aggregation (no momentum effect)
        result1 = aggregator.aggregate(self.model_updates, self.data_sizes)

        # Second aggregation (momentum should be applied)
        new_params_1 = {
            'weight': torch.randn(5, 5),
            'bias': torch.randn(5)
        }
        new_params_2 = {
            'weight': torch.randn(5, 5),
            'bias': torch.randn(5)
        }
        new_updates = [(0, new_params_1), (1, new_params_2)]

        result2 = aggregator.aggregate(new_updates, self.data_sizes)

        # Results should be different due to momentum
        self.assertIsInstance(result2, dict)
        self.assertIn('weight', result2)
        self.assertIn('bias', result2)

    def test_aggregation_stats_enhanced(self):
        """Test enhanced aggregation statistics"""
        aggregator = FedAvgAggregator(
            weighted=True,
            momentum=0.1,
            adaptive_weighting=True
        )

        stats = aggregator.compute_aggregation_stats(self.model_updates, self.data_sizes)

        # Check enhanced stats
        self.assertIn('aggregation_method', stats)
        self.assertIn('momentum', stats)
        self.assertIn('adaptive_weighting', stats)

        self.assertEqual(stats['aggregation_method'], 'fedavg')
        self.assertEqual(stats['momentum'], 0.1)
        self.assertTrue(stats['adaptive_weighting'])

    def test_weighted_vs_unweighted_aggregation(self):
        """Test difference between weighted and unweighted aggregation"""
        # Weighted aggregator
        weighted_agg = FedAvgAggregator(weighted=True, momentum=0.0)
        weighted_result = weighted_agg.aggregate(self.model_updates, self.data_sizes)

        # Unweighted aggregator
        unweighted_agg = FedAvgAggregator(weighted=False, momentum=0.0)
        unweighted_result = unweighted_agg.aggregate(self.model_updates, self.data_sizes)

        # Results should be different when data sizes are different
        for param_name in weighted_result.keys():
            weight_diff = torch.norm(weighted_result[param_name] - unweighted_result[param_name])
            # Should be different (not exactly zero due to different weighting)
            self.assertGreater(weight_diff.item(), 1e-6)


class TestIntegrationFeatures(unittest.TestCase):
    """Test integration of enhanced features"""

    def test_end_to_end_communication_flow(self):
        """Test complete communication flow between multiple nodes"""
        # Create 3 nodes in ring topology
        num_nodes = 3
        protocols = {}

        # Initialize protocols for ring topology
        for i in range(num_nodes):
            neighbors = [(i - 1) % num_nodes, (i + 1) % num_nodes]
            protocols[i] = DFLCommunicationProtocol(
                node_id=i,
                neighbors=neighbors,
                latency_ms=1.0,
                bandwidth_mbps=1000.0
            )

        # Each node broadcasts model update
        model_updates = {}
        for i in range(num_nodes):
            update = {
                'parameters': {
                    'weight': torch.randn(3, 3) + i,  # Different for each node
                    'bias': torch.randn(3) + i
                },
                'data_size': 100 + i * 50,
                'node_id': i
            }
            model_updates[i] = update
            protocols[i].broadcast_to_neighbors({'model_update': update})

        # Each node receives updates from neighbors
        received_updates = {}
        for i in range(num_nodes):
            messages = protocols[i].receive_messages()
            neighbor_updates = {}

            for message in messages:
                sender_id = message['sender_id']
                if 'model_update' in message['content']:
                    neighbor_updates[sender_id] = message['content']['model_update']

            received_updates[i] = neighbor_updates

        # Verify each node received updates from its neighbors
        for i in range(num_nodes):
            neighbors = [(i - 1) % num_nodes, (i + 1) % num_nodes]
            self.assertEqual(len(received_updates[i]), 2)  # Should receive from 2 neighbors

            for neighbor_id in neighbors:
                self.assertIn(neighbor_id, received_updates[i])

        # Clean up
        DFLCommunicationProtocol.reset_global_buffer()

    def test_aggregation_with_communication_data(self):
        """Test aggregation using data from communication simulation"""
        # Simulate received model updates
        model_updates = [
            (0, {'weight': torch.ones(3, 3), 'bias': torch.ones(3)}),
            (1, {'weight': torch.ones(3, 3) * 2, 'bias': torch.ones(3) * 2}),
            (2, {'weight': torch.ones(3, 3) * 3, 'bias': torch.ones(3) * 3})
        ]
        data_sizes = {0: 100, 1: 200, 2: 150}

        # Test aggregation
        aggregator = FedAvgAggregator(weighted=True, momentum=0.1)
        result = aggregator.aggregate(model_updates, data_sizes)

        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn('weight', result)
        self.assertIn('bias', result)

        # Verify weighted average (node 1 has highest weight)
        expected_weight_sum = (100 * 1 + 200 * 2 + 150 * 3) / (100 + 200 + 150)
        actual_weight_mean = result['weight'].mean().item()

        self.assertAlmostEqual(actual_weight_mean, expected_weight_sum, places=2)


if __name__ == '__main__':
    unittest.main()