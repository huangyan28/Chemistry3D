import torch
from torch.utils.data import DataLoader
from inference.models.network import TGCNN
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import glob
from torchvision.transforms import functional as TF
import argparse
import matplotlib.cm as cm
import cv2
import segmentation_models_pytorch as smp

    parser = argparse.ArgumentParser(description='Inference on a folder of images using a trained model')
    parser.add_argument('--test_dir', type=str, default='/home/huangyan/Dataset/Transparent_Dataset/Image',
                        help='Directory containing the test images')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold value for the prediction')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/huangyan/Mission/runs/runs_20240315-221700/tgcnn_best_model.pth',
                        help='Path to the trained model checkpoint')
    parser.add_argument('--output_dir', type=str,
                        default="/home/huangyan/Dataset/Transparent_Dataset/Image/TGC",
                        help='Directory to save the prediction results')
    parser.add_argument('--model_type', type=str, default='TGCNN', choices=['TGCNN', 'Unet', 'UnetPlusPlus', 'DeepLabV3'],
                        help='Type of the model to use for inference')
    parser.add_argument('--encoder_name', type=str, default='resnet34',
                        help='Name of the encoder (for models that require an encoder)')
    parser.add_argument('--encoder_weights', type=str, default='imagenet',
                        help='Pre-trained weights for the encoder')

def save_prediction_output(output, image_name, threshold=0.5, folder="prediction_results"):
    """
    Save the prediction output as an image.
    
    Args:
        output (Tensor): The prediction output.
        image_name (str): The name of the image file.
        threshold (float): Threshold value for the prediction.
        folder (str): Folder to save the prediction results.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    binary_output = (output.cpu().numpy() > threshold).astype(np.int)
    colormap = np.array([[0, 0, 0], [255, 255, 255]])
    colored_output = colormap[binary_output]
    colored_output_image = Image.fromarray(colored_output.astype(np.uint8))
    output_path = os.path.join(folder, image_name)
    colored_output_image.save(output_path)

def inference_single_image(image_path, model_checkpoint, model_type, encoder_name, encoder_weights, output_folder="prediction_results"):
    """
    Perform inference on a single image.
    
    Args:
        image_path (str): Path to the input image.
        model_checkpoint (str): Path to the model checkpoint.
        model_type (str): Type of the model.
        encoder_name (str): Name of the encoder.
        encoder_weights (str): Pre-trained weights for the encoder.
        output_folder (str): Folder to save the prediction results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == 'TGCNN':
        model = TGCNN(input_channels=3, output_channels=1)
    elif model_type == 'Unet':
        model = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=3, classes=1)
    elif model_type == 'UnetPlusPlus':
        model = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=3, classes=1)
    elif model_type == 'DeepLabV3':
        model = smp.DeepLabV3(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=3, classes=1)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.load_state_dict(torch.load(model_checkpoint))
    model = model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = TF.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)

    save_prediction_output(output[0][0], os.path.basename(image_path), output_folder)

def save_prediction_heatmap(output, image_name, folder="prediction_heatmaps"):
    """
    Save the prediction heatmap using OpenCV's applyColorMap.
    
    Args:
        output (Tensor): The prediction output.
        image_name (str): The name of the image file.
        folder (str): Folder to save the heatmaps.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Convert the model output to a NumPy array and normalize to 0-255
    output_np = output.cpu().numpy()
    output_np_normalized = cv2.normalize(output_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # Convert float to uint8
    output_np_uint8 = np.uint8(output_np_normalized)
    
    # Apply heatmap color mapping
    heatmap = cv2.applyColorMap(output_np_uint8, cv2.COLORMAP_JET)
    
    # Construct the output file path
    output_path = os.path.join(folder, image_name)
    
    # Save the heatmap
    cv2.imwrite(output_path, heatmap)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = ChemDataset(image_dir=args.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    if args.model_type == 'TGCNN':
        model = TGCNN(input_channels=3, output_channels=1)
    elif args.model_type == 'Unet':
        model = smp.Unet(encoder_name=args.encoder_name, encoder_weights=args.encoder_weights, in_channels=3, classes=1)
    elif args.model_type == 'UnetPlusPlus':
        model = smp.UnetPlusPlus(encoder_name=args.encoder_name, encoder_weights=args.encoder_weights, in_channels=3, classes=1)
    elif args.model_type == 'DeepLabV3':
        model = smp.DeepLabV3(encoder_name=args.encoder_name, encoder_weights=args.encoder_weights, in_channels=3, classes=1)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    model.load_state_dict(torch.load(args.checkpoint))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for i, (inputs, image_name) in tqdm(enumerate(test_loader), total=len(test_loader)):
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            save_prediction_output(outputs[0][0], image_name[0], folder=os.path.join(args.output_dir))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)