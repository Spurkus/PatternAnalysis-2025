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
from dataset import AlzheimerDataset
from modules import GFNetPyramid
from modules import Mixup


def train_one_epoch(model, data_loader, optimizer, criterion, device, scaler, mixup_fn=None):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0

    progress_bar = tqdm(data_loader, desc="Training", unit="batch")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.float().to(device)

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        # Forward pass with autocast
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(images)
            outputs = outputs.squeeze(1)
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
        images, labels = images.to(device), labels.float().to(device)

        # Forward pass
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(images)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)

        total_loss += loss.item()

        # Calculate accuracy
        predicted = (outputs.data >= 0).float()
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
                    mean=[0.5], std=[0.5]
                ),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5], std=[0.5]
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
    # Split a validation set from the training data (held-out) so we don't evaluate on the test set during training
    full_train_len = len(train_dataset)
    val_len = int(full_train_len * args.val_split)
    train_len = full_train_len - val_len

    if val_len > 0 and train_len > 0:
        # Reproducible split
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_len, val_len], generator=generator
        )

        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        print(f"Training samples: {train_len}, Validation samples: {val_len}")
    else:
        # Fallback: use the entire training set and use the test set as validation (not recommended)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = None
        print("No validation split requested or dataset too small; continuing without a dedicated validation set.")

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
        patch_size=16,
        embed_dim=[64, 128, 256, 512],
        depth=[3, 3, 9, 3],
        mlp_ratio=[4, 4, 4, 4],
        drop_path_rate=0.15,
        num_classes=1,
    ).to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_parameters / 1e6:.2f}M")

    # Loss function, optimizer, and scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_epochs = 10
    print(f"Using {warmup_epochs}-epoch linear warmup + long cosine decay.")

    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6
    )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
    )

    scaler = torch.amp.GradScaler()  # For mixed precision training

    print(f"Initializing Mixup with alpha={args.mixup_alpha} and prob={args.mixup_prob}")
    mixup_fn = Mixup(
        mixup_alpha=args.mixup_alpha,
        prob=args.mixup_prob,
        device=device
    )

    # Training loop
    best_accuracy = 0.0
    start_time = time.time()
    patience = 30
    epochs_no_improve = 0

    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    val_accuracies = []
    learning_rates = []

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, mixup_fn=mixup_fn
        )
        print(f"Epoch {epoch+1} Average Training Loss: {train_loss:.4f}")

        # Use the held-out validation loader during training if available
        if val_loader is None:
            # If no val split was created, fall back to using the test set (not ideal)
            current_eval_loader = test_loader
            print("Warning: No validation split - using test set for validation during training.")
        else:
            current_eval_loader = val_loader

        val_loss, val_accuracy = evaluate(model, current_eval_loader, criterion, device)
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
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(
                f"No improvement for {patience} consecutive epochs. Early stopping..."
            )
            break

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time/60:.2f} minutes.")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")

    # Plot and save metrics
    plot_metrics(
        train_losses, val_losses, val_accuracies, learning_rates, args.output_dir
    )

    # --- Final evaluation on the test set using the best saved model ---
    best_model_path = os.path.join(args.output_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"\nLoading best model from {best_model_path} for final test evaluation...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        print(f"Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_accuracy:.2f}%")
    else:
        print(f"No best model found at {best_model_path}; skipping final test evaluation.")


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
        "--epochs", type=int, default=200, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.8,
        help="Alpha parameter for MixUp. (Default: 0.8)",
    )
    parser.add_argument(
        "--mixup-prob",
        type=float,
        default=1.0,
        help="Probability of applying MixUp. (Default: 1.0)",
    )
    parser.add_argument(
        "--weight-decay", 
        type=float, 
        default=0.05,
        help="Weight decay (L2 penalty)"
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of the training data to hold out for validation (e.g. 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for dataset splitting.",
    )

    args = parser.parse_args()
    main(args)
