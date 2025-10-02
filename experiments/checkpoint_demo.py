"""
Checkpoint Demo - Save/Load Node State để tiếp tục training
"""

import torch
import yaml
import os
import sys
import time
import pickle
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.node import BaseNode
from src.models import create_model
from src.data import DataPartitioner


class CheckpointManager:
    """Quản lý checkpoint cho node"""

    def __init__(self, checkpoint_dir="./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_node_state(self, node, node_name):
        """Save complete node state"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{node_name}_checkpoint.pt")

        state = {
            'node_id': node.node_id,
            'model_state_dict': node.model.state_dict(),
            'optimizer_state_dict': node.optimizer.state_dict(),
            'history': node.history,
            'config': node.config,
            'data_size': node.get_data_size()
        }

        torch.save(state, checkpoint_path)
        print(f"💾 Node state saved to: {checkpoint_path}")
        return checkpoint_path

    def load_node_state(self, node, node_name):
        """Load node state if exists"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{node_name}_checkpoint.pt")

        if not os.path.exists(checkpoint_path):
            print(f"📂 No checkpoint found at: {checkpoint_path}")
            return False

        state = torch.load(checkpoint_path, map_location=node.device)

        node.model.load_state_dict(state['model_state_dict'])
        node.optimizer.load_state_dict(state['optimizer_state_dict'])
        node.history = state['history']

        print(f"📥 Node state loaded from: {checkpoint_path}")
        print(f"   Previous training history: {len(node.history['train_loss'])} epochs")
        return True

    def save_data_partition(self, node_subsets, partition_name):
        """Save data partition để không phải partition lại"""
        partition_path = os.path.join(self.checkpoint_dir, f"{partition_name}_partition.pkl")

        # Convert subsets to indices để save
        partition_data = []
        for subset in node_subsets:
            partition_data.append(subset.indices)

        with open(partition_path, 'wb') as f:
            pickle.dump(partition_data, f)

        print(f"💾 Data partition saved to: {partition_path}")
        return partition_path

    def load_data_partition(self, dataset, partition_name):
        """Load data partition if exists"""
        partition_path = os.path.join(self.checkpoint_dir, f"{partition_name}_partition.pkl")

        if not os.path.exists(partition_path):
            print(f"📂 No partition found at: {partition_path}")
            return None

        with open(partition_path, 'rb') as f:
            partition_indices = pickle.load(f)

        # Recreate subsets
        node_subsets = []
        for indices in partition_indices:
            subset = torch.utils.data.Subset(dataset, indices)
            node_subsets.append(subset)

        print(f"📥 Data partition loaded from: {partition_path}")
        print(f"   {len(node_subsets)} nodes, sizes: {[len(s) for s in node_subsets]}")
        return node_subsets


def create_or_load_node(node_id, checkpoint_manager, device):
    """Tạo node mới hoặc load từ checkpoint"""

    # 1. Load hoặc create data partition
    print("🔄 Checking for existing data partition...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    node_subsets = checkpoint_manager.load_data_partition(train_dataset, "3nodes_noniid")

    if node_subsets is None:
        print("📊 Creating new data partition...")
        partitioner = DataPartitioner(
            dataset=train_dataset, num_nodes=3, alpha=0.5, seed=42
        )
        node_subsets = partitioner.partition_dirichlet()
        checkpoint_manager.save_data_partition(node_subsets, "3nodes_noniid")

    # 2. Create DataLoader
    train_loader = DataLoader(
        node_subsets[node_id], batch_size=64, shuffle=True, num_workers=2
    )

    # 3. Create model
    print("🧠 Creating model...")
    model = create_model(num_classes=10)
    model = model.to(device)

    # 4. Create node
    config = {
        'device': device,
        'learning_rate': 0.01,
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 0.0005
    }

    node = BaseNode(
        node_id=node_id,
        model=model,
        dataloader=train_loader,
        config=config
    )

    # 5. Try to load checkpoint
    loaded = checkpoint_manager.load_node_state(node, f"node_{node_id}")

    if loaded:
        print(f"✅ Node {node_id} restored from checkpoint")
    else:
        print(f"🆕 Node {node_id} created fresh")

    return node


def continue_training(node, checkpoint_manager, epochs=2):
    """Tiếp tục training và save checkpoint"""

    print(f"\n🚀 Continuing training for {epochs} more epochs...")

    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform
    )

    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=2
    )

    # Evaluate before training
    initial_metrics = node.evaluate(test_loader)
    print(f"   Current test accuracy: {initial_metrics['test_accuracy']:.4f}")

    # Train
    start_time = time.time()
    train_metrics = node.local_train(epochs)
    training_time = time.time() - start_time

    print(f"   Training completed in {training_time:.2f}s")
    print(f"   Train loss: {train_metrics['loss']:.4f}")
    print(f"   Train accuracy: {train_metrics['accuracy']:.4f}")

    # Evaluate after training
    final_metrics = node.evaluate(test_loader)
    improvement = final_metrics['test_accuracy'] - initial_metrics['test_accuracy']

    print(f"   New test accuracy: {final_metrics['test_accuracy']:.4f}")
    print(f"   Improvement: {improvement:.4f}")

    # Save checkpoint
    checkpoint_manager.save_node_state(node, f"node_{node.node_id}")

    return {
        'initial_acc': initial_metrics['test_accuracy'],
        'final_acc': final_metrics['test_accuracy'],
        'improvement': improvement,
        'train_time': training_time
    }


def main():
    print("🔄 Checkpoint Demo - Resume Training")
    print("=" * 50)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(42)

    checkpoint_manager = CheckpointManager()

    # Tạo hoặc load node
    node = create_or_load_node(node_id=0, checkpoint_manager=checkpoint_manager, device=device)

    print(f"\n📊 Node Info:")
    print(f"   Node ID: {node.node_id}")
    print(f"   Data size: {node.get_data_size()}")
    print(f"   Training history: {len(node.history['train_loss'])} previous epochs")

    # Training session 1
    print(f"\n{'='*20} TRAINING SESSION {'='*20}")
    results = continue_training(node, checkpoint_manager, epochs=2)

    print(f"\n📈 Session Results:")
    print(f"   Training time: {results['train_time']:.2f}s")
    print(f"   Accuracy improvement: {results['improvement']:.4f}")
    print(f"   Current accuracy: {results['final_acc']:.4f}")

    # Simulate stopping and restarting
    print(f"\n💤 Simulating restart...")
    print("   (Trong thực tế, bạn có thể dừng chương trình và chạy lại)")

    # Load lại node từ checkpoint
    node_reloaded = create_or_load_node(node_id=0, checkpoint_manager=checkpoint_manager, device=device)

    # Training session 2
    print(f"\n{'='*20} RESUMED SESSION {'='*20}")
    results2 = continue_training(node_reloaded, checkpoint_manager, epochs=2)

    print(f"\n🎯 Final Results:")
    print(f"   Total training sessions: 2")
    print(f"   Total epochs: {len(node_reloaded.history['train_loss'])}")
    print(f"   Final accuracy: {results2['final_acc']:.4f}")
    print(f"   Session 2 improvement: {results2['improvement']:.4f}")

    print("\n✅ Checkpoint demo completed!")
    print("💡 Bây giờ bạn có thể dừng và tiếp tục training bất cứ lúc nào!")


if __name__ == "__main__":
    main()