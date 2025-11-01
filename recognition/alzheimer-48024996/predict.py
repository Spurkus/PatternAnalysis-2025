import argparse
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from modules import GFNetPyramid


def predict(args):
    """
    Loads the model and performs a prediction on a single image.
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the same transformations used for validation
    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Initialize the model with the same architecture as during training
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

    # Load the trained model weights
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at '{args.model_path}'")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.eval()

    # Load and preprocess the image
    try:
        image = Image.open(args.image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
    except FileNotFoundError:
        print(f"Error: Image file not found at '{args.image_path}'")
        return
    except Exception as e:
        print(f"Error opening or processing image: {e}")
        return

    # Class names mapping
    class_names = ["Normal Control (NC)", "Alzheimer's Disease (AD)"]

    # Perform prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        # Get the top prediction
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_class = class_names[predicted_idx.item()]

    print("\n--- Prediction Probabilities ---")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {probabilities[i].item() * 100:.2f}%")

    print("\n--- Final Prediction ---")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence.item() * 100:.2f}%")

    if not args.no_plot:
        try:
            plt.figure(figsize=(6, 7))
            plt.imshow(image)
            # Create a title string with the prediction and confidence
            title_str = (
                f"Prediction: {predicted_class}\n"
                f"Confidence: {confidence.item() * 100:.2f}%"
            )
            plt.title(title_str, fontsize=14)
            plt.axis("off")  # Hide axes
            print("\nDisplaying prediction plot... (Close the window to exit)")
            plt.show()
        except Exception as e:
            # Handle cases where GUI is not available (e.g., SSH terminal)
            print(f"\nCould not display plot. Error: {e}")
            print("To disable this, run with the --no-plot flag.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict Alzheimer's Disease from a single brain scan image."
    )
    parser.add_argument(
        "--image-path", type=str, required=True, help="Path to the input image file."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--img-size", type=int, default=224, help="Image size the model was trained on."
    )
    # Added argument to disable plotting
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable showing the image plot visualization.",
    )

    args = parser.parse_args()
    predict(args)
