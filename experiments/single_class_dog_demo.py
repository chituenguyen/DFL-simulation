"""
🐕 Single Class Training Demo - DOG Only
Train a node ONLY on DOG images, then test predictions on DOG and CAT images.
Shows probability distributions to understand model's confidence.
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

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

CAT_CLASS = 3
DOG_CLASS = 5


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


def get_class_predictions(node, test_loader, target_class, num_samples=100):
    """
    Get predictions and probabilities for a specific class

    Args:
        node: BaseNode instance
        test_loader: DataLoader with test data
        target_class: Class to get predictions for
        num_samples: Number of samples to collect

    Returns:
        Dict with predictions and probabilities
    """
    node.model.eval()

    predictions = []
    probabilities = []
    images_collected = []
    labels_collected = []

    with torch.no_grad():
        for images, labels in test_loader:
            # Filter for target class
            mask = (labels == target_class)
            if mask.sum() == 0:
                continue

            class_images = images[mask]
            class_labels = labels[mask]

            # Move to device
            class_images = class_images.to(node.device)

            # Get predictions
            outputs = node.model(class_images)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # Store results
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            images_collected.extend(class_images.cpu().numpy())
            labels_collected.extend(class_labels.numpy())

            if len(predictions) >= num_samples:
                break

    # Trim to num_samples
    predictions = predictions[:num_samples]
    probabilities = probabilities[:num_samples]
    images_collected = images_collected[:num_samples]
    labels_collected = labels_collected[:num_samples]

    return {
        'predictions': np.array(predictions),
        'probabilities': np.array(probabilities),
        'images': np.array(images_collected),
        'labels': np.array(labels_collected)
    }


def analyze_predictions(results, target_class, class_name):
    """Analyze and print prediction statistics"""
    predictions = results['predictions']
    probabilities = results['probabilities']

    # Accuracy
    correct = (predictions == target_class).sum()
    total = len(predictions)
    accuracy = 100 * correct / total

    # Probability statistics for target class
    target_probs = probabilities[:, target_class]

    print(f"\n  📊 {class_name.upper()} Images Analysis:")
    print(f"     Total samples: {total}")
    print(f"     Correctly predicted as {class_name}: {correct}/{total} ({accuracy:.2f}%)")
    print(f"\n     Probability for class '{class_name}':")
    print(f"       Mean:   {target_probs.mean():.4f}")
    print(f"       Median: {np.median(target_probs):.4f}")
    print(f"       Min:    {target_probs.min():.4f}")
    print(f"       Max:    {target_probs.max():.4f}")
    print(f"       Std:    {target_probs.std():.4f}")

    # Show most confident predictions
    top_indices = np.argsort(target_probs)[-5:][::-1]
    print(f"\n     Top 5 confident predictions as '{class_name}':")
    for i, idx in enumerate(top_indices, 1):
        pred_class = predictions[idx]
        pred_name = CIFAR10_CLASSES[pred_class]
        prob = target_probs[idx]
        print(f"       {i}. Predicted: {pred_name} (prob={prob:.4f})")

    # Show average probability for each class
    print(f"\n     Average probability distribution across all classes:")
    avg_probs = probabilities.mean(axis=0)
    for class_id, prob in enumerate(avg_probs):
        bar = '█' * int(prob * 50)
        print(f"       {CIFAR10_CLASSES[class_id]:12s}: {prob:.4f} |{bar}")


def visualize_predictions(dog_results, cat_results, save_path='results/plots/single_class_dog_predictions.png'):
    """Visualize sample predictions"""
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle('🐕🐱 Predictions: Node Trained ONLY on DOG', fontsize=16, fontweight='bold')

    # Denormalize images
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    def denormalize(img):
        img = img.transpose(1, 2, 0)
        img = img * std + mean
        img = np.clip(img, 0, 1)
        return img

    # Top row: DOG images (5 samples)
    for i in range(5):
        ax = axes[0, i]
        img = denormalize(dog_results['images'][i])
        ax.imshow(img)

        pred = dog_results['predictions'][i]
        prob = dog_results['probabilities'][i][pred]
        dog_prob = dog_results['probabilities'][i][DOG_CLASS]

        title = f"True: DOG\nPred: {CIFAR10_CLASSES[pred]}\nProb: {prob:.3f}\nDOG: {dog_prob:.3f}"
        color = 'green' if pred == DOG_CLASS else 'red'
        ax.set_title(title, fontsize=9, color=color, fontweight='bold')
        ax.axis('off')

    # Second row: DOG probability bars
    for i in range(5):
        ax = axes[1, i]
        probs = dog_results['probabilities'][i]
        colors = ['orange' if j == DOG_CLASS else 'lightblue' for j in range(10)]
        ax.bar(range(10), probs, color=colors)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(10))
        ax.set_xticklabels([c[:3] for c in CIFAR10_CLASSES], rotation=45, fontsize=7)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
        ax.set_ylabel('Prob', fontsize=8)
        if i == 0:
            ax.set_title('Probability Distribution', fontsize=9, fontweight='bold')

    # Third row: CAT images (5 samples)
    for i in range(5):
        ax = axes[2, i]
        img = denormalize(cat_results['images'][i])
        ax.imshow(img)

        pred = cat_results['predictions'][i]
        prob = cat_results['probabilities'][i][pred]
        cat_prob = cat_results['probabilities'][i][CAT_CLASS]

        title = f"True: CAT\nPred: {CIFAR10_CLASSES[pred]}\nProb: {prob:.3f}\nCAT: {cat_prob:.3f}"
        color = 'green' if pred == CAT_CLASS else 'red'
        ax.set_title(title, fontsize=9, color=color, fontweight='bold')
        ax.axis('off')

    # Fourth row: CAT probability bars
    for i in range(5):
        ax = axes[3, i]
        probs = cat_results['probabilities'][i]
        colors = ['green' if j == CAT_CLASS else 'lightblue' for j in range(10)]
        ax.bar(range(10), probs, color=colors)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(10))
        ax.set_xticklabels([c[:3] for c in CIFAR10_CLASSES], rotation=45, fontsize=7)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
        ax.set_ylabel('Prob', fontsize=8)
        if i == 0:
            ax.set_title('Probability Distribution', fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Create directory if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  💾 Visualization saved to: {save_path}")

    # Also try to show
    try:
        plt.show()
    except:
        pass


def main():
    print_header("🐕 SINGLE CLASS TRAINING DEMO - DOG ONLY")

    # Setup
    print_section("⚙️  Setup")
    device = torch.device("mps" if torch.backends.mps.is_available()
                         else "cuda" if torch.cuda.is_available()
                         else "cpu")
    print(f"  Device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)
    print(f"  Random seed: 42")

    # Load data
    print_section("📊 Loading CIFAR-10 Data")
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

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    # Partition data - only DOG class
    print_section("🔀 Data Partitioning - DOG ONLY")
    partitioner = DataPartitioner(
        dataset=train_dataset,
        num_nodes=1,
        alpha=0.5,
        seed=42
    )

    # Use class-based partitioning to get only DOG
    dog_subset = partitioner.partition_class_based([[DOG_CLASS]])[0]

    print(f"  Node data: {len(dog_subset)} samples (only DOG class)")

    # Create DataLoaders
    train_loader = DataLoader(
        dog_subset,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2
    )

    # Create model and node
    print_section("🧠 Creating Model and Node")
    model = create_model(num_classes=10)

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

    print(f"  Model: ResNet-18")
    print(f"  Optimizer: SGD (lr=0.01, momentum=0.9)")

    # Initial evaluation (before training)
    print_section("📊 BEFORE Training - Random Weights")

    print("  Getting predictions on test set...")
    dog_results_before = get_class_predictions(node, test_loader, DOG_CLASS, num_samples=100)
    cat_results_before = get_class_predictions(node, test_loader, CAT_CLASS, num_samples=100)

    analyze_predictions(dog_results_before, DOG_CLASS, 'dog')
    analyze_predictions(cat_results_before, CAT_CLASS, 'cat')

    # Training
    print_section("🚀 Training on DOG Images Only")
    num_epochs = 10

    for epoch in range(1, num_epochs + 1):
        metrics = node.local_train(epochs=1)
        print(f"  Epoch {epoch}/{num_epochs} | "
              f"Loss: {metrics['loss']:.4f} | "
              f"Accuracy: {metrics['accuracy']*100:.2f}%")

    print("\n  ✅ Training completed!")

    # After training evaluation
    print_section("📊 AFTER Training on DOG")

    print("  Getting predictions on test set...")
    dog_results_after = get_class_predictions(node, test_loader, DOG_CLASS, num_samples=100)
    cat_results_after = get_class_predictions(node, test_loader, CAT_CLASS, num_samples=100)

    analyze_predictions(dog_results_after, DOG_CLASS, 'dog')
    analyze_predictions(cat_results_after, CAT_CLASS, 'cat')

    # Compare before and after
    print_section("📈 Comparison: BEFORE vs AFTER Training")

    dog_prob_before = dog_results_before['probabilities'][:, DOG_CLASS].mean()
    dog_prob_after = dog_results_after['probabilities'][:, DOG_CLASS].mean()

    cat_prob_before = cat_results_before['probabilities'][:, CAT_CLASS].mean()
    cat_prob_after = cat_results_after['probabilities'][:, CAT_CLASS].mean()

    print(f"\n  🐕 DOG Images (True class: DOG):")
    print(f"     Avg probability for DOG class:")
    print(f"       Before training: {dog_prob_before:.4f}")
    print(f"       After training:  {dog_prob_after:.4f}")
    print(f"       Improvement:     +{dog_prob_after - dog_prob_before:.4f}")

    print(f"\n  🐱 CAT Images (True class: CAT, but model never saw CAT):")
    print(f"     Avg probability for CAT class:")
    print(f"       Before training: {cat_prob_before:.4f}")
    print(f"       After training:  {cat_prob_after:.4f}")
    print(f"       Change:          {cat_prob_after - cat_prob_before:+.4f}")

    # What does model predict for CAT images?
    cat_preds_after = cat_results_after['predictions']
    unique, counts = np.unique(cat_preds_after, return_counts=True)

    print(f"\n  🤔 What does model predict for CAT images?")
    for class_id, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:5]:
        percentage = 100 * count / len(cat_preds_after)
        print(f"     {CIFAR10_CLASSES[class_id]:12s}: {count:3d} times ({percentage:.1f}%)")

    # Visualize
    print_section("🎨 Creating Visualization")
    visualize_predictions(dog_results_after, cat_results_after)

    print_header("✅ Demo Completed!")

    print("\n  🎯 Key Findings:")
    print(f"  • Model trained ONLY on DOG images ({len(dog_subset)} samples)")
    print(f"  • On DOG images: probability increased from {dog_prob_before:.4f} to {dog_prob_after:.4f}")
    print(f"  • On CAT images: model has NOT learned to recognize CAT")
    print(f"  • Model often confuses CAT with other classes (never saw CAT during training)")
    print(f"  • This shows why Federated Learning is important - to share knowledge!")
    print()


if __name__ == "__main__":
    main()
