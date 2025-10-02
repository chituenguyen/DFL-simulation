"""
Single Node Training Demo - Hiểu luồng chạy từng bước
"""

import torch
import yaml
import os
import sys
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.node import BaseNode
from src.models import create_model
from src.data import DataPartitioner


def main():
    print("🚀 Single Node Training Demo")
    print("=" * 50)

    # 1. Setup cơ bản
    print("📋 Step 1: Basic setup...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   Device: {device}")

    torch.manual_seed(42)
    print("   Random seed: 42")

    # 2. Load dữ liệu
    print("\n📊 Step 2: Loading CIFAR-10 data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=False,
        transform=transform
    )

    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")

    # 3. Phân chia dữ liệu cho 1 node (giả lập có 3 nodes, lấy phần của node 0)
    print("\n🔀 Step 3: Data partitioning...")
    partitioner = DataPartitioner(
        dataset=train_dataset,
        num_nodes=3,
        alpha=0.5,
        seed=42
    )

    node_subsets = partitioner.partition_dirichlet()
    node_0_subset = node_subsets[0]  # Lấy data của node 0

    print(f"   Node 0 gets: {len(node_0_subset)} samples")

    # 4. Tạo DataLoader
    print("\n📦 Step 4: Creating DataLoaders...")
    # node_0_subset đã là Subset rồi, không cần tạo lại

    train_loader = DataLoader(
        node_0_subset,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # 5. Tạo model
    print("\n🧠 Step 5: Creating model...")
    model = create_model(num_classes=10)
    model = model.to(device)

    print(f"   Model: ResNet18")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 6. Tạo Node config
    print("\n🏗️ Step 6: Creating Node...")
    config = {
        'device': device,
        'learning_rate': 0.01,
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 0.0005
    }

    node = BaseNode(
        node_id=0,
        model=model,
        dataloader=train_loader,
        config=config
    )

    print(f"   Node ID: {node.node_id}")
    print(f"   Data size: {node.get_data_size()}")

    # 7. Training
    print("\n🚀 Step 7: Starting training...")

    # Đánh giá trước training
    print("   Evaluating before training...")
    initial_metrics = node.evaluate(test_loader)
    print(f"   Initial accuracy: {initial_metrics['test_accuracy']:.4f}")

    # Training 2 epochs
    local_epochs = 2
    print(f"   Training for {local_epochs} epochs...")

    start_time = time.time()
    train_metrics = node.local_train(local_epochs)
    training_time = time.time() - start_time

    print(f"   Training completed in {training_time:.2f}s")
    print(f"   Final train loss: {train_metrics['loss']:.4f}")
    print(f"   Final train accuracy: {train_metrics['accuracy']:.4f}")

    # 8. Evaluation sau training
    print("\n📈 Step 8: Final evaluation...")
    final_metrics = node.evaluate(test_loader)
    print(f"   Final test accuracy: {final_metrics['test_accuracy']:.4f}")
    print(f"   Final test loss: {final_metrics['test_loss']:.4f}")

    # 9. So sánh
    print("\n📊 Step 9: Results comparison...")
    improvement = final_metrics['test_accuracy'] - initial_metrics['test_accuracy']
    print(f"   Accuracy improvement: {improvement:.4f}")

    print("\n✅ Single node training demo completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()