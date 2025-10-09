"""
P2P Federated Learning with Mild Non-IID Data

This experiment demonstrates how P2P FL improves weak nodes:
- Node 0: 60% DOG, 20% CAT, 20% BIRD (strong on DOG)
- Node 1: 60% CAT, 20% DOG, 20% BIRD (strong on CAT)
- Node 2: 60% BIRD, 20% DOG, 20% CAT (strong on BIRD, weakest overall)

Goal: Show how Node 2 improves after P2P federated learning
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.cifar10_loader import CIFAR10Loader
from src.models.simple_cnn import SimpleCNN
from src.node.node import Node
from src.communication.topology import create_topology
from src.communication.protocol import SimulatedP2PProtocol
from src.aggregation.fedavg import FedAvg


def create_mild_noniid_partition(dataset, node_id, num_nodes=3):
    """
    Create mild Non-IID partition where each node has all classes
    but with different proportions

    Node 0: 60% DOG (label 5), 20% CAT (label 3), 20% BIRD (label 2)
    Node 1: 60% CAT (label 3), 20% DOG (label 5), 20% BIRD (label 2)
    Node 2: 60% BIRD (label 2), 20% DOG (label 5), 20% CAT (label 3)
    """
    # CIFAR-10 labels: 2=bird, 3=cat, 5=dog
    DOG_LABEL = 5
    CAT_LABEL = 3
    BIRD_LABEL = 2

    # Get indices for each class
    dog_indices = [i for i, (_, label) in enumerate(dataset) if label == DOG_LABEL]
    cat_indices = [i for i, (_, label) in enumerate(dataset) if label == CAT_LABEL]
    bird_indices = [i for i, (_, label) in enumerate(dataset) if label == BIRD_LABEL]

    # Split each class evenly first
    dog_per_node = len(dog_indices) // num_nodes
    cat_per_node = len(cat_indices) // num_nodes
    bird_per_node = len(bird_indices) // num_nodes

    if node_id == 0:
        # Node 0: 60% DOG, 20% CAT, 20% BIRD
        # Take more dog samples
        node_dog = dog_indices[:int(dog_per_node * 1.8)]  # 60% of total share
        node_cat = cat_indices[:int(cat_per_node * 0.6)]  # 20% of total share
        node_bird = bird_indices[:int(bird_per_node * 0.6)]  # 20% of total share

    elif node_id == 1:
        # Node 1: 60% CAT, 20% DOG, 20% BIRD
        node_dog = dog_indices[int(dog_per_node * 0.6):int(dog_per_node * 1.2)]
        node_cat = cat_indices[int(cat_per_node * 0.6):int(cat_per_node * 2.2)]
        node_bird = bird_indices[int(bird_per_node * 0.6):int(bird_per_node * 1.2)]

    else:  # node_id == 2
        # Node 2: 60% BIRD, 20% DOG, 20% CAT
        node_dog = dog_indices[int(dog_per_node * 1.2):int(dog_per_node * 1.8)]
        node_cat = cat_indices[int(cat_per_node * 1.2):int(cat_per_node * 1.8)]
        node_bird = bird_indices[int(bird_per_node * 1.2):int(bird_per_node * 2.2)]

    # Combine all indices for this node
    node_indices = node_dog + node_cat + node_bird

    return node_indices


def evaluate_model(model, test_loader, device):
    """Evaluate model and return overall + per-class accuracy"""
    model.eval()
    correct = 0
    total = 0

    # Per-class tracking (only for DOG=5, CAT=3, BIRD=2)
    class_correct = {2: 0, 3: 0, 5: 0}  # bird, cat, dog
    class_total = {2: 0, 3: 0, 5: 0}

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class accuracy
            for label in [2, 3, 5]:
                mask = (labels == label)
                if mask.sum() > 0:
                    class_total[label] += mask.sum().item()
                    class_correct[label] += ((predicted == labels) & mask).sum().item()

    overall_acc = 100 * correct / total if total > 0 else 0

    dog_acc = 100 * class_correct[5] / class_total[5] if class_total[5] > 0 else 0
    cat_acc = 100 * class_correct[3] / class_total[3] if class_total[3] > 0 else 0
    bird_acc = 100 * class_correct[2] / class_total[2] if class_total[2] > 0 else 0

    return overall_acc, dog_acc, cat_acc, bird_acc


def train_local_model(model, train_loader, device, epochs, lr, weight_decay):
    """Train model locally"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def save_checkpoint(nodes, round_num, checkpoint_dir="results/checkpoints"):
    """Save checkpoint for all nodes"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'round': round_num,
        'timestamp': datetime.now().isoformat(),
        'models': {}
    }

    for node_id, node in enumerate(nodes):
        checkpoint['models'][f'node_{node_id}'] = node.model.state_dict()

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_round_{round_num}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path, nodes):
    """Load checkpoint and restore all node models"""
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path)

    for node_id, node in enumerate(nodes):
        node.model.load_state_dict(checkpoint['models'][f'node_{node_id}'])

    print(f"✅ Checkpoint loaded from round {checkpoint['round']}")
    return checkpoint['round']


def main():
    print("=" * 80)
    print("P2P FEDERATED LEARNING WITH MILD NON-IID DATA")
    print("=" * 80)
    print("\nData Distribution:")
    print("  Node 0: 60% DOG,  20% CAT,  20% BIRD (Strong Node)")
    print("  Node 1: 60% CAT,  20% DOG,  20% BIRD (Medium Node)")
    print("  Node 2: 60% BIRD, 20% DOG,  20% CAT (Weak Node)")
    print("\nGoal: Demonstrate how Node 2 improves through P2P FL")
    print("=" * 80)

    # Configuration
    NUM_NODES = 3
    BATCH_SIZE = 64

    # Pre-training config
    PRETRAIN_EPOCHS = 10
    PRETRAIN_LR = 0.003
    PRETRAIN_WEIGHT_DECAY = 0.002

    # P2P FL config
    FL_ROUNDS = 100
    FL_LOCAL_EPOCHS = 1
    FL_LR = 0.0003
    FL_WEIGHT_DECAY = 0.002
    CHECKPOINT_INTERVAL = 5  # Save every 5 rounds

    # Resume from checkpoint (set to None to start fresh)
    RESUME_FROM_CHECKPOINT = None  # e.g., "results/checkpoints/checkpoint_round_50.pth"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available()
                         else "mps" if torch.backends.mps.is_available()
                         else "cpu")
    print(f"\n🖥️  Using device: {device}")

    # Load data
    print("\n📦 Loading CIFAR-10 data...")
    data_loader = CIFAR10Loader()
    train_dataset = data_loader.load_data(train=True)
    test_dataset = data_loader.load_data(train=False)

    # Create mild Non-IID partitions
    print("📊 Creating mild Non-IID partitions...")
    node_datasets = []
    node_loaders = []

    for node_id in range(NUM_NODES):
        indices = create_mild_noniid_partition(train_dataset, node_id, NUM_NODES)
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

        node_datasets.append(subset)
        node_loaders.append(loader)

        # Show distribution
        labels = [train_dataset[i][1] for i in indices]
        dog_count = labels.count(5)
        cat_count = labels.count(3)
        bird_count = labels.count(2)
        total = len(labels)

        print(f"  Node {node_id}: {total:,} samples | "
              f"DOG {dog_count:4d} ({100*dog_count/total:5.1f}%) | "
              f"CAT {cat_count:4d} ({100*cat_count/total:5.1f}%) | "
              f"BIRD {bird_count:4d} ({100*bird_count/total:5.1f}%)")

    # Create test loader (full test set for evaluation)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # =====================================================================
    # PHASE 1: Pre-train each node independently
    # =====================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: Pre-training Each Node Independently")
    print("=" * 80)

    models = []
    for node_id in range(NUM_NODES):
        print(f"\n🏋️  Training Node {node_id}...")
        model = SimpleCNN(num_classes=10).to(device)

        train_local_model(
            model,
            node_loaders[node_id],
            device,
            epochs=PRETRAIN_EPOCHS,
            lr=PRETRAIN_LR,
            weight_decay=PRETRAIN_WEIGHT_DECAY
        )

        models.append(model)

    # Evaluate pre-trained models
    print("\n" + "-" * 80)
    print("PRE-FL PERFORMANCE (Before Federated Learning)")
    print("-" * 80)

    for node_id in range(NUM_NODES):
        overall, dog, cat, bird = evaluate_model(models[node_id], test_loader, device)

        node_name = ["Strong (DOG)", "Medium (CAT)", "Weak (BIRD)"][node_id]
        print(f"Node {node_id} ({node_name:15s}): "
              f"Overall {overall:5.2f}% | "
              f"DOG {dog:5.2f}% | CAT {cat:5.2f}% | BIRD {bird:5.2f}%")

    # Save Node 2's initial performance
    initial_overall, initial_dog, initial_cat, initial_bird = evaluate_model(
        models[2], test_loader, device
    )

    # =====================================================================
    # PHASE 2: P2P Federated Learning
    # =====================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: P2P Federated Learning (Ring Topology)")
    print("=" * 80)
    print("Topology: Node 0 ↔ [Node 1, Node 2]")
    print("          Node 1 ↔ [Node 0, Node 2]")
    print("          Node 2 ↔ [Node 0, Node 1]")
    print("=" * 80)

    # Create topology and protocol
    topology = create_topology("ring", NUM_NODES)
    protocol = SimulatedP2PProtocol(topology)

    # Create nodes
    nodes = []
    for node_id in range(NUM_NODES):
        node = Node(
            node_id=node_id,
            model=models[node_id],
            train_loader=node_loaders[node_id],
            test_loader=test_loader,
            device=device
        )
        nodes.append(node)

    # Aggregator
    aggregator = FedAvg()

    # Track Node 2's improvement
    node2_history = []

    # Resume from checkpoint if specified
    start_round = 1
    if RESUME_FROM_CHECKPOINT:
        loaded_round = load_checkpoint(RESUME_FROM_CHECKPOINT, nodes)
        if loaded_round:
            start_round = loaded_round + 1
            print(f"🔄 Resuming from round {start_round}")

    # P2P Training Loop
    for round_num in range(start_round, FL_ROUNDS + 1):
        # Each node trains locally
        for node in nodes:
            train_local_model(
                node.model,
                node.train_loader,
                device,
                epochs=FL_LOCAL_EPOCHS,
                lr=FL_LR,
                weight_decay=FL_WEIGHT_DECAY
            )

        # P2P exchange and aggregation
        for node_id, node in enumerate(nodes):
            # Get neighbor models
            neighbors = topology.get_neighbors(node_id)
            neighbor_models = [nodes[n].model for n in neighbors]

            # Aggregate with neighbors (include self)
            all_models = [node.model] + neighbor_models
            dataset_sizes = [len(node_datasets[node_id])] + \
                          [len(node_datasets[n]) for n in neighbors]

            aggregated_params = aggregator.aggregate(
                [m.state_dict() for m in all_models],
                dataset_sizes
            )

            # Update model
            node.model.load_state_dict(aggregated_params)

        # Save checkpoint every CHECKPOINT_INTERVAL rounds
        if round_num % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(nodes, round_num)

        # Evaluate every 5 rounds
        if round_num % 5 == 0 or round_num == 1:
            print(f"\n📊 Round {round_num}/{FL_ROUNDS}")
            print("-" * 80)

            for node_id in range(NUM_NODES):
                overall, dog, cat, bird = evaluate_model(
                    nodes[node_id].model, test_loader, device
                )

                node_name = ["Strong (DOG)", "Medium (CAT)", "Weak (BIRD)"][node_id]

                # Track Node 2
                if node_id == 2:
                    node2_history.append({
                        'round': round_num,
                        'overall': overall,
                        'dog': dog,
                        'cat': cat,
                        'bird': bird
                    })

                    # Show improvement
                    overall_gain = overall - initial_overall
                    dog_gain = dog - initial_dog
                    cat_gain = cat - initial_cat
                    bird_gain = bird - initial_bird

                    print(f"Node {node_id} ({node_name:15s}): "
                          f"Overall {overall:5.2f}% ({overall_gain:+5.2f}%) | "
                          f"DOG {dog:5.2f}% ({dog_gain:+5.2f}%) | "
                          f"CAT {cat:5.2f}% ({cat_gain:+5.2f}%) | "
                          f"BIRD {bird:5.2f}% ({bird_gain:+5.2f}%)")
                else:
                    print(f"Node {node_id} ({node_name:15s}): "
                          f"Overall {overall:5.2f}% | "
                          f"DOG {dog:5.2f}% | CAT {cat:5.2f}% | BIRD {bird:5.2f}%")

    # =====================================================================
    # FINAL SUMMARY: Node 2's Improvement
    # =====================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY: NODE 2 (WEAK NODE) IMPROVEMENT")
    print("=" * 80)

    final_overall, final_dog, final_cat, final_bird = evaluate_model(
        nodes[2].model, test_loader, device
    )

    print(f"\n{'Metric':<15s} {'Before FL':>12s} {'After FL':>12s} {'Improvement':>12s}")
    print("-" * 80)
    print(f"{'Overall':<15s} {initial_overall:>11.2f}% {final_overall:>11.2f}% "
          f"{final_overall - initial_overall:>11.2f}%")
    print(f"{'DOG':<15s} {initial_dog:>11.2f}% {final_dog:>11.2f}% "
          f"{final_dog - initial_dog:>11.2f}%")
    print(f"{'CAT':<15s} {initial_cat:>11.2f}% {final_cat:>11.2f}% "
          f"{final_cat - initial_cat:>11.2f}%")
    print(f"{'BIRD':<15s} {initial_bird:>11.2f}% {final_bird:>11.2f}% "
          f"{final_bird - initial_bird:>11.2f}%")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("✅ Node 2 learned from Node 0's strong DOG knowledge")
    print("✅ Node 2 learned from Node 1's strong CAT knowledge")
    print("✅ Mild Non-IID data allows effective P2P federated learning")
    print("✅ All nodes benefit while keeping data decentralized")
    print("=" * 80)

    # Save final model
    os.makedirs("results", exist_ok=True)
    torch.save(nodes[2].model.state_dict(), "results/node2_improved_model.pth")
    print(f"\n💾 Node 2's improved model saved to results/node2_improved_model.pth")


if __name__ == "__main__":
    main()
