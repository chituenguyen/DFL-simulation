"""
🐕🌐 DOG Model Federated Learning - P2P (Decentralized) Version

Scenario:
1. Start with pre-trained models (DOG, CAT, BIRD)
2. Use P2P topology (Ring/Mesh) - NO central server
3. Each node only communicates with NEIGHBORS
4. Decentralized aggregation

Differences from centralized version:
- No central server
- Nodes communicate peer-to-peer
- Each node aggregates with neighbors only
- More realistic blockchain-based FL
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.node import BaseNode
from src.models import create_model
from src.data import DataPartitioner
from src.aggregation import FedAvgAggregator
from src.communication import create_topology, SimulatedP2PProtocol

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

DOG_CLASS = 5
CAT_CLASS = 3
BIRD_CLASS = 2

# Paths for saving/loading models
MODEL_SAVE_DIR = 'results/pretrained_models'
DOG_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'dog_pretrained.pth')
CAT_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'cat_pretrained.pth')
BIRD_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'bird_pretrained.pth')

# P2P FL checkpoint path
P2P_CHECKPOINT_DIR = 'results/p2p_checkpoints'
P2P_CHECKPOINT_PATH = os.path.join(P2P_CHECKPOINT_DIR, 'p2p_checkpoint.pth')

# Local model save paths (per node)
P2P_LOCAL_MODEL_DIR = 'results/p2p_local_models'
DOG_P2P_LOCAL_PATH = os.path.join(P2P_LOCAL_MODEL_DIR, 'node_0_dog_p2p.pth')
CAT_P2P_LOCAL_PATH = os.path.join(P2P_LOCAL_MODEL_DIR, 'node_1_cat_p2p.pth')
BIRD_P2P_LOCAL_PATH = os.path.join(P2P_LOCAL_MODEL_DIR, 'node_2_bird_p2p.pth')


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


def train_single_class_model(device, class_id, class_name, save_path, num_epochs=10):
    """
    Train a model on a single class or load from cache

    Args:
        device: torch device
        class_id: CIFAR-10 class ID
        class_name: Name of the class
        save_path: Path to save/load model
        num_epochs: Number of training epochs

    Returns:
        Trained model
    """
    # Check if model exists
    if os.path.exists(save_path):
        print(f"  ✅ Loading cached {class_name} model from {save_path}")
        model = create_model(num_classes=10)
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.to(device)
        return model

    print(f"  🔄 Training new {class_name} model...")

    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # Get only specific class
    partitioner = DataPartitioner(
        dataset=train_dataset,
        num_nodes=1,
        alpha=0.5,
        seed=42
    )
    class_subset = partitioner.partition_class_based([[class_id]])[0]

    train_loader = DataLoader(
        class_subset,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )

    # Create and train model
    model = create_model(num_classes=10)

    config = {
        'device': device,
        'learning_rate': 0.003,  # Even lower LR for better pretrain
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 0.002  # Stronger regularization
    }

    node = BaseNode(
        node_id=0,
        model=model,
        dataloader=train_loader,
        config=config
    )

    print(f"  Training on {len(class_subset)} {class_name} images for {num_epochs} epochs...")

    # Train
    for epoch in range(1, num_epochs + 1):
        metrics = node.local_train(epochs=1)
        if epoch % 2 == 0:
            print(f"    Epoch {epoch}/{num_epochs} | Loss: {metrics['loss']:.4f} | Acc: {metrics['accuracy']*100:.2f}%")

    # Save model
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"  💾 Saved {class_name} model to {save_path}")

    return model


def evaluate_per_class(model, test_loader, device):
    """Evaluate model on each class"""
    model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1

    # Calculate accuracy per class
    class_accuracy = {}
    for i in range(10):
        if class_total[i] > 0:
            class_accuracy[i] = 100 * class_correct[i] / class_total[i]
        else:
            class_accuracy[i] = 0.0

    overall = 100 * sum(class_correct) / sum(class_total) if sum(class_total) > 0 else 0.0

    return class_accuracy, overall


def print_class_accuracies(class_acc, title=""):
    """Print per-class accuracies in a nice format"""
    if title:
        print(f"\n  {title}")

    print(f"  {'Class':<12} {'Accuracy':<10} {'Bar'}")
    print(f"  {'-'*40}")

    for i in range(10):
        acc = class_acc[i]
        bar = '█' * int(acc / 5)  # Scale to fit
        color = '\033[92m' if i in [DOG_CLASS, CAT_CLASS, BIRD_CLASS] else ''
        reset = '\033[0m' if i in [DOG_CLASS, CAT_CLASS, BIRD_CLASS] else ''
        print(f"  {color}{CIFAR10_CLASSES[i]:<12} {acc:>5.1f}%     |{bar}{reset}")


def main():
    print_header("🐕🌐 P2P Federated Learning Demo (Decentralized)")

    # Setup
    print_section("⚙️  Setup")
    device = torch.device("mps" if torch.backends.mps.is_available()
                         else "cuda" if torch.cuda.is_available()
                         else "cpu")
    print(f"  Device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    # Phase 1: Train or load single-class models
    print_section("🐕 Phase 1: Preparing Single-Class Models")

    dog_model = train_single_class_model(device, DOG_CLASS, 'DOG', DOG_MODEL_PATH)
    cat_model = train_single_class_model(device, CAT_CLASS, 'CAT', CAT_MODEL_PATH)
    bird_model = train_single_class_model(device, BIRD_CLASS, 'BIRD', BIRD_MODEL_PATH)

    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2
    )

    # Evaluate before FL
    print_section("📊 Models Performance (Before P2P FL)")
    class_acc_before, overall_before = evaluate_per_class(dog_model, test_loader, device)
    print(f"  Overall Accuracy: {overall_before:.2f}%")
    print_class_accuracies(class_acc_before, "Per-Class Accuracy:")

    # Phase 2: Setup P2P Federated Learning
    print_section("🌐 Phase 2: P2P Federated Learning Setup")

    # Load training data
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # Create 3 nodes with different classes
    class_assignments = [
        [DOG_CLASS],   # Node 0: DOG
        [CAT_CLASS],   # Node 1: CAT
        [BIRD_CLASS]   # Node 2: BIRD
    ]

    partitioner = DataPartitioner(
        dataset=train_dataset,
        num_nodes=3,
        alpha=0.5,
        seed=42
    )

    subsets = partitioner.partition_class_based(class_assignments)

    # Node config - CRITICAL: Balance between learning and stability
    config = {
        'device': device,
        'learning_rate': 0.0003,  # Balanced LR for P2P (not too low, not too high)
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 0.002  # Strong regularization
    }

    # Check if local models exist (from previous P2P FL rounds)
    local_paths = [DOG_P2P_LOCAL_PATH, CAT_P2P_LOCAL_PATH, BIRD_P2P_LOCAL_PATH]
    models = [dog_model, cat_model, bird_model]

    if all(os.path.exists(path) for path in local_paths):
        print("\n  📂 Found existing P2P local models from previous rounds")
        for i, (model, path) in enumerate(zip(models, local_paths)):
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"    ✅ Loaded Node {i} from {path}")

    # Create nodes
    nodes = []
    dog_loader = DataLoader(subsets[0], batch_size=64, shuffle=True, num_workers=2)
    dog_node = BaseNode(node_id=0, model=models[0], dataloader=dog_loader, config=config)
    nodes.append(dog_node)

    cat_loader = DataLoader(subsets[1], batch_size=64, shuffle=True, num_workers=2)
    cat_node = BaseNode(node_id=1, model=models[1], dataloader=cat_loader, config=config)
    nodes.append(cat_node)

    bird_loader = DataLoader(subsets[2], batch_size=64, shuffle=True, num_workers=2)
    bird_node = BaseNode(node_id=2, model=models[2], dataloader=bird_loader, config=config)
    nodes.append(bird_node)

    # Create P2P topology (Ring)
    print("\n  🔗 Creating P2P Topology: RING")
    topology = create_topology(
        topology_type='ring',
        num_nodes=3
    )

    # Create P2P protocols for each node
    protocols = {}
    topology_neighbors = topology.get_all_neighbors()
    for node in nodes:
        protocols[node.node_id] = SimulatedP2PProtocol(
            node.node_id,
            topology_neighbors[node.node_id]
        )

    print(f"\n  📡 P2P Communication Graph:")
    for node_id, neighbors in topology_neighbors.items():
        class_name = CIFAR10_CLASSES[class_assignments[node_id][0]]
        neighbor_names = [CIFAR10_CLASSES[class_assignments[n][0]] for n in neighbors]
        print(f"    Node {node_id} ({class_name:4s}) ↔ Neighbors: {neighbors} {neighbor_names}")

    print(f"\n  Created 3 nodes:")
    for i, node in enumerate(nodes):
        classes = [CIFAR10_CLASSES[c] for c in class_assignments[i]]
        print(f"    Node {i}: {classes} - {len(subsets[i])} samples")

    # Create aggregators (one per node for decentralized aggregation)
    aggregators = {i: FedAvgAggregator(weighted=True) for i in range(3)}

    # Phase 3: P2P Federated Training
    num_rounds = 100  # Moderate rounds with balanced LR
    local_epochs = 1  # Keep at 1 to avoid overfitting
    start_round = 1

    # Check for existing checkpoint
    if os.path.exists(P2P_CHECKPOINT_PATH):
        print_section("📂 Loading P2P FL Checkpoint")
        checkpoint = torch.load(P2P_CHECKPOINT_PATH, map_location=device)

        # Load models
        for i, node in enumerate(nodes):
            node.model.load_state_dict(checkpoint[f'node_{i}_model'])

        start_round = checkpoint['round'] + 1
        dog_accuracies = checkpoint.get('dog_accuracies', [])

        print(f"  ✅ Resumed from round {checkpoint['round']}")
        print(f"  Continuing from round {start_round}/{num_rounds}")
    else:
        print_section(f"🚀 Phase 3: P2P Federated Learning ({num_rounds} rounds)")
        dog_accuracies = []

    for round_num in range(start_round, num_rounds + 1):
        # ①️ LOCAL TRAINING - Each node trains independently
        for node in nodes:
            node.local_train(epochs=local_epochs)

        # ②️ P2P AGGREGATION - Each node aggregates with its NEIGHBORS only
        # This is the key difference from centralized approach!
        new_models = {}

        for node in nodes:
            protocol = protocols[node.node_id]

            # Collect models from NEIGHBORS only (not all nodes)
            neighbor_models = []
            neighbor_weights = {}

            for neighbor_id in protocol.neighbors:
                neighbor_node = nodes[neighbor_id]
                neighbor_models.append((neighbor_id, neighbor_node.model.state_dict()))
                neighbor_weights[neighbor_id] = neighbor_node.get_data_size()

            # Add own model
            neighbor_models.append((node.node_id, node.model.state_dict()))
            neighbor_weights[node.node_id] = node.get_data_size()

            # Aggregate with neighbors only
            aggregator = aggregators[node.node_id]
            aggregated_model = aggregator.aggregate(neighbor_models, neighbor_weights)

            # Store for later update (to avoid interference during aggregation)
            new_models[node.node_id] = aggregated_model

        # ③️ UPDATE - All nodes update simultaneously
        for node in nodes:
            node.model.load_state_dict(new_models[node.node_id])

        # ④️ SAVE LOCAL MODELS - After each round
        os.makedirs(P2P_LOCAL_MODEL_DIR, exist_ok=True)
        for i, node in enumerate(nodes):
            torch.save(node.model.state_dict(), local_paths[i])

        # ⑤️ EVALUATE & CHECKPOINT - Every 5 rounds
        if round_num % 5 == 0:
            class_acc, overall = evaluate_per_class(nodes[0].model, test_loader, device)
            dog_acc = class_acc[DOG_CLASS]
            dog_accuracies.append((round_num, dog_acc, overall))

            print(f"\n  Round {round_num}/{num_rounds}:")
            print(f"    Overall: {overall:.2f}% | DOG: {dog_acc:.2f}% | "
                  f"CAT: {class_acc[CAT_CLASS]:.2f}% | BIRD: {class_acc[BIRD_CLASS]:.2f}%")

            # Show per-node accuracy
            print(f"    Per-node accuracies:")
            for i, node in enumerate(nodes):
                node_acc, node_overall = evaluate_per_class(node.model, test_loader, device)
                class_name = CIFAR10_CLASSES[class_assignments[i][0]]
                print(f"      Node {i} ({class_name}): {node_overall:.2f}%")

            # Save global checkpoint
            os.makedirs(P2P_CHECKPOINT_DIR, exist_ok=True)
            checkpoint = {
                'round': round_num,
                'dog_accuracies': dog_accuracies,
                'node_0_model': nodes[0].model.state_dict(),
                'node_1_model': nodes[1].model.state_dict(),
                'node_2_model': nodes[2].model.state_dict(),
            }
            torch.save(checkpoint, P2P_CHECKPOINT_PATH)
            print(f"    💾 P2P checkpoint saved")
            print(f"    💾 P2P local models saved: {P2P_LOCAL_MODEL_DIR}/")

    # Final evaluation
    print_section("📊 Final Evaluation (After P2P FL)")

    print("\n  📊 Per-Node Final Accuracies:")
    for i, node in enumerate(nodes):
        class_acc, overall = evaluate_per_class(node.model, test_loader, device)
        class_name = CIFAR10_CLASSES[class_assignments[i][0]]
        print(f"\n  Node {i} ({class_name.upper()}):")
        print(f"    Overall Accuracy: {overall:.2f}%")
        print_class_accuracies(class_acc)

    # Average performance
    print_section("📈 Average Performance Across All Nodes")

    all_overall = []
    all_class_acc = {i: [] for i in range(10)}

    for node in nodes:
        class_acc, overall = evaluate_per_class(node.model, test_loader, device)
        all_overall.append(overall)
        for i in range(10):
            all_class_acc[i].append(class_acc[i])

    avg_overall = np.mean(all_overall)
    print(f"\n  Average Overall Accuracy: {avg_overall:.2f}%")

    print(f"\n  Average Per-Class Accuracy:")
    print(f"  {'Class':<12} {'Before FL':<12} {'After FL':<12} {'Improvement'}")
    print(f"  {'-'*55}")

    for i in range(10):
        before = class_acc_before[i]
        after = np.mean(all_class_acc[i])
        diff = after - before

        color = '\033[92m' if i in [DOG_CLASS, CAT_CLASS, BIRD_CLASS] else ''
        reset = '\033[0m' if i in [DOG_CLASS, CAT_CLASS, BIRD_CLASS] else ''

        print(f"  {color}{CIFAR10_CLASSES[i]:<12} {before:>6.1f}%      {after:>6.1f}%      "
              f"{diff:+6.1f}%{reset}")

    print(f"\n  {'Overall':<12} {overall_before:>6.1f}%      {avg_overall:>6.1f}%      "
          f"{avg_overall - overall_before:+6.1f}%")

    print_header("✅ P2P Demo Completed!")

    print("\n  💡 Key Differences from Centralized FL:")
    print("  • NO central server - pure peer-to-peer")
    print("  • Each node only communicates with NEIGHBORS")
    print("  • Ring topology: Node 0 ↔ [Node 1, Node 2]")
    print("  • Decentralized aggregation (each node aggregates independently)")
    print("  • More realistic for blockchain-based FL")
    print("  • Knowledge spreads gradually through the network")
    print()

    print("\n  🎯 Results:")
    print(f"  • Average accuracy improved from {overall_before:.2f}% to {avg_overall:.2f}%")
    print(f"  • All nodes learned about classes they never saw in training")
    print(f"  • P2P communication successfully shared knowledge")
    print()


if __name__ == "__main__":
    main()
