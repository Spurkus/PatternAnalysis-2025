import os
from torch.utils.data import Dataset
from typing import Optional, Callable
from PIL import Image


class AlzheimerDataset(Dataset):
    """
    Dataset class for Alzheimer's disease recognition.
    """

    def __init__(
        self,
        root_dir: str = "AD_NC",
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the dataset.

        Parameters:
            root_dir: Path to the root directory
            split: The dataset split, 'train' or 'test'.
            transform: Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.data_path = os.path.join(self.root_dir, self.split)

        # Define the classes and mapping
        self.classes = ["NC", "AD"]  # NC: Normal Control, AD: Alzheimer's Disease
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = self._load_samples()

    def _load_samples(self) -> list[tuple[str, int]]:
        """
        Loads all image paths and their corresponding labels.

        Returns:
            list of tuples: Each tuple contains (image_path, label)
        """
        samples = []

        # Iterate through each class directory ('NC', 'AD')
        for target_class in self.classes:
            label = self.class_to_idx[target_class]
            class_dir = os.path.join(self.data_path, target_class)

            # Check if the directory exists
            if not os.path.isdir(class_dir):
                print(f"Warning: Directory not found: {class_dir}")
                continue

            # Walk through the directory and find all .jpeg files
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(".jpeg") or filename.lower().endswith(
                    ".jpg"
                ):
                    path = os.path.join(class_dir, filename)
                    item = (path, label)
                    samples.append(item)

        return samples

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        """
        Fetches a data sample for a given index.

        Parameters:
            idx: The index of the sample to fetch.

        Returns:
            A tuple (image, label) where image is the transformed image and label is its class index.
        """
        # Get the image path and label for the given index
        img_path, label = self.samples[idx]

        # Open the image using Pillow
        image = Image.open(img_path).convert("L")

        # Apply transformations if they are provided
        if self.transform:
            image = self.transform(image)

        return image, label
