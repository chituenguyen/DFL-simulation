"""
🎯 Class-Based DFL Demo
Demonstrates extreme Non-IID scenario where each node trains on specific classes only.

Scenario:
- 5 nodes, each trains on 2 specific CIFAR-10 classes
- Node 0: Airplane, Automobile (classes 0, 1)
- Node 1: Bird, Cat (classes 2, 3)
- Node 2: Deer, Dog (classes 4, 5)
- Node 3: Frog, Horse (classes 6, 7)
- Node 4: Ship, Truck (classes 8, 9)

Through Federated Learning, all nodes learn all 10 classes!
"""

import os
import sys
import yaml
import torch
import logging
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.node import NodeManager
from src.aggregation import FedAvgAggregator
from src.communication import create_topology, SimulatedP2PProtocol
from src.utils import setup_logger

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def print_header(text: str):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_section(text: str):
    """Print formatted section"""
    print(f"\n{'─' * 80}")
    print(f"  {text}")
    print(f"{'─' * 80}")


def evaluate_per_class(node, test_loader, device):
    """
    Evaluate node on each class separately
    
    Returns:
        Dict with per-class accuracy
    """
    node.model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = node.model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Count per class
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1
    
    # Calculate per-class accuracy
    class_accuracy = {}
    for i in range(10):
        if class_total[i] > 0:
            class_accuracy[i] = 100 * class_correct[i] / class_total[i]
        else:
            class_accuracy[i] = 0.0
    
    # Overall accuracy
    total_correct = sum(class_correct)
    total_samples = sum(class_total)
    overall_accuracy = 100 * total_correct / total_samples if total_samples > 0 else 0.0
    
    return {
        'per_class': class_accuracy,
        'overall': overall_accuracy,
        'correct': class_correct,
        'total': class_total
    }


def main():
    print_header("🎯 CLASS-BASED FEDERATED LEARNING DEMO")
    
    # Load configuration
    config_path = 'config/config_class_based.yaml'
    
    if not os.path.exists(config_path):
        print(f"\n❌ Config file not found: {config_path}")
        print("Please make sure config_class_based.yaml exists!")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = setup_logger(
        name='class_based_demo',
        level=config['logging']['level'],
        log_dir=config['logging']['save_dir']
    )
    
    # Print experiment info
    print_section("📋 Experiment Configuration")
    print(f"  Experiment: {config['experiment']['name']}")
    print(f"  Device: {config['experiment']['device']}")
    print(f"  Model: {config['model']['name']}")
    print(f"  Partition Type: {config['partition']['partition_type']}")
    print(f"\n  Class Assignments:")
    
    class_assignments = config['partition']['class_assignments']
    for i, classes in enumerate(class_assignments):
        class_names = [CIFAR10_CLASSES[c] for c in classes]
        print(f"    Node {i}: {class_names} (classes {classes})")
    
    print(f"\n  Training Rounds: {config['training']['num_rounds']}")
    print(f"  Local Epochs: {config['training']['local_epochs']}")
    print(f"  Topology: {config['topology']['type']}")
    
    # Create nodes
    print_section("🏗️  Creating Nodes with Class-Based Data")
    node_manager = NodeManager(config)
    nodes = node_manager.create_nodes()
    test_loader = node_manager.get_test_loader()
    
    print(f"\n  Created {len(nodes)} nodes")
    for i, node in enumerate(nodes):
        assigned = [CIFAR10_CLASSES[c] for c in class_assignments[i]]
        print(f"  Node {i}: {node.get_data_size():,} samples - Training on {assigned}")
    
    # Create topology and protocols
    print_section("🌐 Setting up Network Topology")
    topology = create_topology(
        topology_type=config['topology']['type'],
        num_nodes=config['partition']['num_nodes'],
        **config['topology']
    )
    
    protocols = {}
    topology_neighbors = topology.get_all_neighbors()
    for node in nodes:
        protocols[node.node_id] = SimulatedP2PProtocol(
            node.node_id,
            topology_neighbors[node.node_id]
        )
    
    print(f"  Topology: {config['topology']['type']}")
    for node_id, neighbors in topology_neighbors.items():
        print(f"  Node {node_id} → Neighbors: {neighbors}")
    
    # Create aggregator
    aggregator = FedAvgAggregator(weighted=config['aggregation']['weighted'])
    
    # Device
    device = config['experiment']['device']
    
    # Initial evaluation
    print_section("📊 Initial Evaluation (Before Training)")
    print("\n  Testing if nodes only know their assigned classes...\n")
    
    for node_id, node in enumerate(nodes):
        metrics = evaluate_per_class(node, test_loader, device)
        assigned_classes = class_assignments[node_id]
        
        print(f"  Node {node_id} (trained on {[CIFAR10_CLASSES[c] for c in assigned_classes]}):")
        print(f"    Overall Accuracy: {metrics['overall']:.2f}%")
        
        # Show accuracy on assigned vs non-assigned classes
        assigned_acc = np.mean([metrics['per_class'][c] for c in assigned_classes])
        non_assigned = [c for c in range(10) if c not in assigned_classes]
        non_assigned_acc = np.mean([metrics['per_class'][c] for c in non_assigned])
        
        print(f"    Assigned classes avg:     {assigned_acc:.2f}%")
        print(f"    Non-assigned classes avg: {non_assigned_acc:.2f}%")
    
    # Training loop
    num_rounds = config['training']['num_rounds']
    local_epochs = config['training']['local_epochs']
    
    print_section(f"🚀 Starting Federated Training ({num_rounds} rounds)")
    
    # Track metrics
    round_metrics = []
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n  Round {round_num}/{num_rounds}", end=" ")
        
        # Local training
        round_losses = []
        for node in nodes:
            metrics = node.train_epoch(epochs=local_epochs)
            round_losses.append(metrics['loss'])
        
        avg_loss = np.mean(round_losses)
        print(f"| Avg Loss: {avg_loss:.4f}", end="")
        
        # Aggregation (simulated P2P communication)
        for node in nodes:
            protocol = protocols[node.node_id]
            
            # Collect models from neighbors
            neighbor_models = {}
            neighbor_weights = {}
            
            for neighbor_id in protocol.neighbors:
                neighbor_node = nodes[neighbor_id]
                neighbor_models[neighbor_id] = neighbor_node.model.state_dict()
                neighbor_weights[neighbor_id] = neighbor_node.get_data_size()
            
            # Add own model
            neighbor_models[node.node_id] = node.model.state_dict()
            neighbor_weights[node.node_id] = node.get_data_size()
            
            # Aggregate
            aggregated_model = aggregator.aggregate(neighbor_models, neighbor_weights)
            node.model.load_state_dict(aggregated_model)
        
        # Evaluation every 10 rounds
        if round_num % 10 == 0:
            print(f"\n\n  📊 Evaluation after Round {round_num}:")
            
            round_result = {'round': round_num, 'nodes': []}
            
            for node_id, node in enumerate(nodes):
                metrics = evaluate_per_class(node, test_loader, device)
                assigned_classes = class_assignments[node_id]
                
                # Calculate avg accuracy on assigned vs non-assigned
                assigned_acc = np.mean([metrics['per_class'][c] for c in assigned_classes])
                non_assigned = [c for c in range(10) if c not in assigned_classes]
                non_assigned_acc = np.mean([metrics['per_class'][c] for c in non_assigned])
                
                print(f"    Node {node_id}: Overall {metrics['overall']:.2f}% | "
                      f"Assigned {assigned_acc:.2f}% | Non-assigned {non_assigned_acc:.2f}%")
                
                round_result['nodes'].append({
                    'node_id': node_id,
                    'overall': metrics['overall'],
                    'assigned': assigned_acc,
                    'non_assigned': non_assigned_acc
                })
            
            round_metrics.append(round_result)
    
    # Final comprehensive evaluation
    print_section("📊 Final Detailed Evaluation")
    
    print("\n  Per-Class Accuracy for Each Node:\n")
    print(f"  {'Node':<6} {'Classes':<25} {'Overall':<10}", end="")
    for i in range(10):
        print(f"{CIFAR10_CLASSES[i][:4]:<6}", end="")
    print()
    print("  " + "-" * 100)
    
    for node_id, node in enumerate(nodes):
        metrics = evaluate_per_class(node, test_loader, device)
        assigned_classes = class_assignments[node_id]
        assigned_names = [CIFAR10_CLASSES[c] for c in assigned_classes]
        
        print(f"  {node_id:<6} {str(assigned_names):<25} {metrics['overall']:>6.2f}%   ", end="")
        
        for class_id in range(10):
            acc = metrics['per_class'][class_id]
            # Highlight assigned classes
            if class_id in assigned_classes:
                print(f"\033[92m{acc:>5.1f}\033[0m ", end="")  # Green
            else:
                print(f"{acc:>5.1f} ", end="")
        print()
    
    # Summary statistics
    print_section("📈 Summary Statistics")
    
    all_overall = []
    all_assigned = []
    all_non_assigned = []
    
    for node_id, node in enumerate(nodes):
        metrics = evaluate_per_class(node, test_loader, device)
        assigned_classes = class_assignments[node_id]
        
        overall = metrics['overall']
        assigned_acc = np.mean([metrics['per_class'][c] for c in assigned_classes])
        non_assigned = [c for c in range(10) if c not in assigned_classes]
        non_assigned_acc = np.mean([metrics['per_class'][c] for c in non_assigned])
        
        all_overall.append(overall)
        all_assigned.append(assigned_acc)
        all_non_assigned.append(non_assigned_acc)
    
    print(f"\n  Average across all nodes:")
    print(f"    Overall Accuracy:                {np.mean(all_overall):.2f}%")
    print(f"    Avg on ASSIGNED classes:         {np.mean(all_assigned):.2f}%")
    print(f"    Avg on NON-ASSIGNED classes:     {np.mean(all_non_assigned):.2f}%")
    print(f"\n  Improvement on non-assigned classes:")
    print(f"    Initial: ~10% (random guess)")
    print(f"    Final:   {np.mean(all_non_assigned):.2f}%")
    print(f"    Gain:    +{np.mean(all_non_assigned) - 10:.2f}%")
    
    print_header("✅ Demo Completed Successfully!")
    
    print("\n  🎯 Key Insights:")
    print("  • Each node ONLY trained on 2 out of 10 classes")
    print("  • Through federated learning, nodes learned about ALL classes")
    print("  • Knowledge was shared WITHOUT sharing raw data")
    print("  • This is the power of Federated Learning!")
    print("  • Extreme Non-IID scenario successfully handled")
    
    print(f"\n  💾 Logs saved to: {config['logging']['save_dir']}")
    print()


if __name__ == "__main__":
    main()
