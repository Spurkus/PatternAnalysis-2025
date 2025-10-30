"""
Originally written by Yongming Rao, Wenliang Zhao, Zheng Zhu, Jiwen Lu, Jie Zhou;
and modified for Alzheimer's Disease recognition.

Utilises the training and evaluation framework as described in:
https://github.com/raoyongming/GFNet/blob/master/main_gfnet.py
https://github.com/raoyongming/GFNet/blob/master/engine.py
"""

import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast

from dataset import AlzheimerDataset
from modules import GFNetPyramid


def train_one_epoch(model, data_loader, optimizer, criterion, device, scaler):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0

    progress_bar = tqdm(data_loader, desc="Training", unit="batch")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass with autocast
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()

        # Scale the loss and call backward()
        scaler.scale(loss).backward()

        # Unscales gradients and calls optimizer.step()
        scaler.step(optimizer)

        # Updates the scale for next iteration
        scaler.update()
        # ---

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(data_loader)


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(data_loader, desc="Evaluating", unit="batch")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        accuracy = 100 * correct_predictions / total_samples
        progress_bar.set_postfix(loss=loss.item(), accuracy=f"{accuracy:.2f}%")

    avg_loss = total_loss / len(data_loader)
    avg_accuracy = 100 * correct_predictions / total_samples
    return avg_loss, avg_accuracy


def plot_metrics(train_losses, val_losses, val_accuracies, learning_rates, output_dir):
    """
    Saves plots for training/validation loss and validation accuracy.
    """
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, "b-", label="Training Loss")
    plt.plot(epochs, val_losses, "r-", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(output_dir, "loss_plot.png")
    plt.savefig(loss_plot_path)
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_accuracies, "g-", label="Validation Accuracy")
    plt.title("Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    acc_plot_path = os.path.join(output_dir, "accuracy_plot.png")
    plt.savefig(acc_plot_path)
    plt.close()

    # Plot Learning Rate
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, learning_rates, "m-", label="Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True)
    lr_plot_path = os.path.join(output_dir, "learning_rate_plot.png")
    plt.savefig(lr_plot_path)
    plt.close()

    print(f"\nMetrics plots saved to {output_dir}")


def main(args):
    """
    Main function to set up data, model, and start training and evaluation.
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transformations
    # Using standard ImageNet normalization as a starting point
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    }

    # Create datasets
    print("Loading datasets...")
    train_dataset = AlzheimerDataset(
        root_dir=args.data_dir, split="train", transform=data_transforms["train"]
    )
    test_dataset = AlzheimerDataset(
        root_dir=args.data_dir, split="test", transform=data_transforms["test"]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print("Datasets loaded successfully.")

    # Initialize the model
    # Using the GFNetPyramid configuration as a default
    print("Initializing GFNetPyramid model...")
    model = GFNetPyramid(
        img_size=args.img_size,
        patch_size=4,
        embed_dim=[64, 128, 256, 512],
        depth=[3, 3, 10, 3],
        mlp_ratio=[4, 4, 4, 4],
        drop_path_rate=0.1,
        num_classes=2,  # Binary classification: NC vs AD
    ).to(device)

    model = torch.compile(model, mode="max-autotune")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_parameters / 1e6:.2f}M")

    # Loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

    # Training loop
    best_accuracy = 0.0
    start_time = time.time()

    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    val_accuracies = []
    learning_rates = []

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        print(f"Epoch {epoch+1} Average Training Loss: {train_loss:.4f}")

        val_loss, val_accuracy = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
        )

        scheduler.step()

        # Append metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        learning_rates.append(optimizer.param_groups[0]["lr"])

        # Save the best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            print(f"New best accuracy: {best_accuracy:.2f}%. Saving model...")
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            torch.save(
                model.state_dict(), os.path.join(args.output_dir, "best_model.pth")
            )

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time/60:.2f} minutes.")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")

    # Plot and save metrics
    plot_metrics(
        train_losses, val_losses, val_accuracies, learning_rates, args.output_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GFNet for Alzheimer's Disease Classification"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="AD_NC",
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints and plots.",
    )
    parser.add_argument("--img-size", type=int, default=224, help="Input image size.")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")

    args = parser.parse_args()
    main(args)
