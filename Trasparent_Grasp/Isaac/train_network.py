import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import segmentation_models_pytorch as smp
import argparse
from inference.models.network import TGCNN
from utils.data.dataset import ChemDataset

# Create an argparse parser
parser = argparse.ArgumentParser(description='Train a segmentation model on ChemDataset')

# Add command-line arguments
parser.add_argument('--image_dir', type=str, default='/home/huangyan/Dataset/Transparent_Dataset/Image_V6', help='Path to the image directory')
parser.add_argument('--label_dir', type=str, default='/home/huangyan/Dataset/Transparent_Dataset/Label_V6', help='Path to the label directory')
parser.add_argument('--pretrained_weights', type=str, default='', help='Path to the pre-trained model weights')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and evaluation')
parser.add_argument('--learning_rate', type=float, default=0.005, help='Initial learning rate')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
parser.add_argument('--model_save_path', type=str, default='runs', help='Path to save trained models and logs')
parser.add_argument('--encoder_name', type=str, default='resnet34', help='Name of the encoder architecture')
parser.add_argument('--encoder_weights', type=str, default='imagenet', help='Pre-trained weights for the encoder')
parser.add_argument('--model_type', type=str, default='Unet', choices=['Unet', 'FPN', 'Linknet', 'PSPNet'], help='Type of the segmentation model')
parser.add_argument('--loss_function', type=str, default='BCEWithLogitsLoss', choices=['BCEWithLogitsLoss', 'CrossEntropyLoss'], help='Loss function to use')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='Optimizer to use')
parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR', choices=['CosineAnnealingLR', 'StepLR'], help='Learning rate scheduler to use')

def parse_args():
    """
    Parse command-line arguments.
    """
    return parser.parse_args()

def setup_environment(args):
    """
    Set up the environment for training.
    
    Args:
        args (Namespace): Parsed command-line arguments.
    
    Returns:
        SummaryWriter: TensorBoard summary writer.
        device (torch.device): Device to be used for training.
    """
    # Ensure the save directory exists
    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(os.path.join(args.model_save_path, 'model'), exist_ok=True)

    # Create SummaryWriter instance for TensorBoard logging
    writer = SummaryWriter(args.model_save_path)

    # Log hyperparameters
    writer.add_scalar('Parameters/Epoch', args.num_epochs)
    writer.add_scalar('Parameters/Initial_LR', args.learning_rate)
    writer.add_scalar('Parameters/Batch_Size', args.batch_size)

    # Set the device for training (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return writer, device

def save_sample_images(inputs, labels, outputs, epoch, idx, folder="validation_samples"):
    """
    Save sample images during validation.

    Args:
        inputs (torch.Tensor): Input images.
        labels (torch.Tensor): True labels.
        outputs (torch.Tensor): Predicted labels.
        epoch (int): Current epoch number.
        idx (int): Current index within the batch.
        folder (str): Directory to save the sample images.
    """
    os.makedirs(folder, exist_ok=True)
    inputs = inputs.cpu().numpy()
    labels = labels.cpu().numpy() * 255
    outputs = (outputs > 0.5).float().cpu().numpy() * 255  # Apply threshold and convert to 0 and 255

    for i in range(inputs.shape[0]):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(np.transpose(inputs[i], (1, 2, 0)))
        axs[0].set_title("Input Image")
        axs[1].imshow(labels[i].squeeze(), cmap='gray', vmin=0, vmax=255)
        axs[1].set_title("True Label")
        axs[2].imshow(outputs[i].squeeze(), cmap='gray', vmin=0, vmax=255)
        axs[2].set_title("Predicted Label")
        
        plt.savefig(f"{folder}/sample_{epoch}_{idx}_{i}.png")
        plt.close()

def evaluate_model_on_test_set(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set.

    Args:
        model (nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader for the test set.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to perform the evaluation on.
    
    Returns:
        float: Average loss over the test set.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(1).float()  # Add a dimension
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(test_loader)

def get_loss_function(name):
    """
    Get the loss function by name.

    Args:
        name (str): Name of the loss function.
    
    Returns:
        nn.Module: Loss function.
    """
    if name == 'BCEWithLogitsLoss':
        return smp.losses.BCEWithLogitsLoss()
    elif name == 'CrossEntropyLoss':
        return smp.losses.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {name}")

def get_optimizer(name, parameters, lr):
    """
    Get the optimizer by name.

    Args:
        name (str): Name of the optimizer.
        parameters (iterable): Model parameters to optimize.
        lr (float): Learning rate.
    
    Returns:
        torch.optim.Optimizer: Optimizer.
    """
    if name == 'Adam':
        return optim.Adam(parameters, lr=lr)
    elif name == 'SGD':
        return optim.SGD(parameters, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

def get_scheduler(name, optimizer, num_epochs):
    """
    Get the learning rate scheduler by name.

    Args:
        name (str): Name of the scheduler.
        optimizer (torch.optim.Optimizer): Optimizer to schedule.
        num_epochs (int): Number of epochs.
    
    Returns:
        torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler.
    """
    if name == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    elif name == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        raise ValueError(f"Unsupported scheduler: {name}")

# Run function encapsulating the training loop
def run():
    args = parse_args()
    
    writer, device = setup_environment(args)

    # Create dataset and split into training and testing sets
    chem_dataset = ChemDataset(
        image_dir=args.image_dir,
        label_dir=args.label_dir
    )

    train_size = int(0.85 * len(chem_dataset))
    test_size = len(chem_dataset) - train_size
    train_dataset, test_dataset = random_split(chem_dataset, [train_size, test_size])

    # Create DataLoader for training and testing datasets
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Initialize the segmentation model based on the selected model type
    if args.model_type == 'Unet':
        model = smp.Unet(encoder_name=args.encoder_name, encoder_weights=args.encoder_weights, in_channels=3, classes=1).to(device)
    elif args.model_type == 'FPN':
        model = smp.FPN(encoder_name=args.encoder_name, encoder_weights=args.encoder_weights, in_channels=3, classes=1).to(device)
    elif args.model_type == 'Linknet':
        model = smp.Linknet(encoder_name=args.encoder_name, encoder_weights=args.encoder_weights, in_channels=3, classes=1).to(device)
    elif args.model_type == 'PSPNet':
        model = smp.PSPNet(encoder_name=args.encoder_name, encoder_weights=args.encoder_weights, in_channels=3, classes=1).to(device)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # Load pre-trained weights if provided
    if args.pretrained_weights:
        model.load_state_dict(torch.load(args.pretrained_weights, map_location=device))
        print(f"Loaded pre-trained weights from {args.pretrained_weights}")

    # Select loss function, optimizer, and scheduler
    criterion = get_loss_function(args.loss_function)
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.learning_rate)
    scheduler = get_scheduler(args.scheduler, optimizer, args.num_epochs)

    # Initialize variables to track the best model
    best_loss = float('inf')
    best_epoch = -1

    # Training loop
    for epoch in tqdm(range(args.num_epochs)):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in tqdm(enumerate(train_loader), desc="Batches", total=len(train_loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(1).float()  # Add a dimension
        
            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate and print the average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {avg_loss:.4f}")
        
        # Log training loss
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        # Evaluate the model on the test set
        test_loss = evaluate_model_on_test_set(model, test_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Test Loss: {test_loss:.4f}")
        
        # Log test loss
        writer.add_scalar('Loss/test', test_loss, epoch)

        # Save the current model
        torch.save(model.state_dict(), os.path.join(args.model_save_path, f"model/{args.model_type.lower()}_model_{epoch}.pth"))

        # Check if the model performs the best on the test set
        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.model_save_path, f"{args.model_type.lower()}_best_model.pth"))
            print(f"Best model saved at epoch {epoch}")
        
        # Update the learning rate
        scheduler.step()

    # Close the SummaryWriter
    writer.close()

    # Save the final model
    torch.save(model.state_dict(), os.path.join(args.model_save_path, f"{args.model_type.lower()}_model_final.pth"))

# Main function
if __name__ == "__main__":
    run()
