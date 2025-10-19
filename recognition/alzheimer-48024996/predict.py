import argparse
import torch
from PIL import Image
from torchvision import transforms

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
        patch_size=4,
        embed_dim=[64, 128, 256, 512],
        depth=[3, 3, 10, 3],
        mlp_ratio=[4, 4, 4, 4],
        drop_path_rate=0.1,
        num_classes=2,
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

    print("\n--- Prediction Result ---")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence.item() * 100:.2f}%")


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

    args = parser.parse_args()
    predict(args)
