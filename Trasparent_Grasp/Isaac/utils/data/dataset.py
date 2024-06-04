import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as TF

def get_transforms():
    """
    Get data transformations for augmentation.
    
    Returns:
        transforms.Compose: Composed transformations for data augmentation.
    """
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
        transforms.ToTensor(),
    ])

class ChemDataset(Dataset):
    """
    Custom Dataset for loading chemical images and their corresponding labels.
    """

    def __init__(self, image_dir, label_dir=None, transform=get_transforms(), image_only=False):
        """
        Initialize the dataset with directories and optional transformations.

        Args:
            image_dir (str): Directory with all the images.
            label_dir (str, optional): Directory with all the labels (numpy arrays). Required if image_only is False.
            transform (callable, optional): Optional transform to be applied on an image.
            image_only (bool): Whether to load images only or both images and labels.
        """
        self.transform = transform
        self.image_dir = image_dir
        self.image_only = image_only
        self.images = sorted(glob.glob(os.path.join(image_dir, "*.png")))

        if not image_only:
            self.label_dir = label_dir
            self.labels = sorted(glob.glob(os.path.join(label_dir, "*.npy")))
            assert len(self.images) == len(self.labels), (
                f"Number of images and labels do not match: "
                f"Number of images = {len(self.images)}, Number of labels = {len(self.labels)}"
            )

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieve an image and optionally its corresponding label by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) if image_only is False, otherwise (image, image_name).
        """
        image_path = self.images[idx]
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # Load image
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        else:
            # Convert image to tensor
            image = TF.to_tensor(image)

        if self.image_only:
            return image
        else:
            label_path = self.labels[idx]
            label = np.load(label_path)

            assert image_name == os.path.splitext(os.path.basename(label_path))[0], (
                f"Image filename '{image_name}' and label filename '{os.path.basename(label_path)}' do not match."
            )

            label = torch.from_numpy(label).long()
            return image, label
