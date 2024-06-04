import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
from matplotlib import colors
import argparse

# Define the color map for class labels
color_map = {
    0: (0, 0, 0),      # Color for class 0
    1: (255, 255, 255),    # Color for class 1
}

def rgb_to_class_id(image, color_map):
    """
    Convert an RGB image to a single-channel class ID image.

    Args:
        image (PIL.Image or np.ndarray): An image in RGB format.
        color_map (dict): A dictionary mapping class IDs to RGB tuples.

    Returns:
        np.ndarray: A single-channel image where each pixel value represents the class ID.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
        
    class_id_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    color_to_class_id = {v: k for k, v in color_map.items()}
    
    for color, class_id in color_to_class_id.items():
        matches = (image == color).all(axis=-1)
        class_id_image[matches] = class_id
        
    return class_id_image

def read_images(folder_path):
    """
    Read all images in a folder and return a list of image paths.

    Args:
        folder_path (str): Path to the folder containing images.

    Returns:
        list: A list of file paths to the images.
    """
    images_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return images_paths

def calculate_metrics_single_class(true_image, pred_image):
    """
    Calculate IoU, Pixel Accuracy (PA), F1-score, and F2-score for a single-class segmentation task.

    Args:
        true_image (np.ndarray): Ground truth class ID image.
        pred_image (np.ndarray): Predicted class ID image.

    Returns:
        tuple: IoU, PA, F1-score, and F2-score.
    """
    true_positive = np.sum((true_image == 1) & (pred_image == 1))
    false_positive = np.sum((true_image == 0) & (pred_image == 1))
    false_negative = np.sum((true_image == 1) & (pred_image == 0))
    true_negative = np.sum((true_image == 0) & (pred_image == 0))
    
    iou = true_positive / (true_positive + false_positive + false_negative)
    pa = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f2_score = 5 * (precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
    
    return iou, pa, f1_score, f2_score

def save_iou_distribution(iou_list, title='IOU Distribution', save_path='iou_distribution.png'):
    """
    Save a plot of IoU distribution.

    Args:
        iou_list (list): List of IoU values.
        title (str): Title of the plot.
        save_path (str): File path to save the plot.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(iou_list, label='IOU per Image')
    plt.axhline(y=np.mean(iou_list), color='r', linestyle='-', label='Mean IOU')
    plt.xlabel('Image Index')
    plt.ylabel('IOU')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def save_pa_distribution(pa_list, title='PA Distribution', save_path='pa_distribution.png'):
    """
    Save a plot of Pixel Accuracy (PA) distribution.

    Args:
        pa_list (list): List of PA values.
        title (str): Title of the plot.
        save_path (str): File path to save the plot.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(pa_list, label='PA per Image')
    plt.axhline(y=np.mean(pa_list), color='r', linestyle='-', label='Mean PA')
    plt.xlabel('Image Index')
    plt.ylabel('PA')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def visualize_metrics(iou_list, pa_list):
    """
    Visualize and save the IoU and PA metrics.

    Args:
        iou_list (list): List of IoU values.
        pa_list (list): List of PA values.
    """
    save_iou_distribution(iou_list)
    save_pa_distribution(pa_list)
    print("Metrics visualization saved")

def process_folders(true_folder, pred_folder):
    """
    Process the ground truth and predicted images to compute and visualize metrics.

    Args:
        true_folder (str): Path to the folder containing ground truth images.
        pred_folder (str): Path to the folder containing predicted images.
    """
    true_images_paths = read_images(true_folder)

    iou_list = []
    pa_list = []
    f1_list = []
    f2_list = []

    for true_path in tqdm(true_images_paths, desc="Processing Images"):
        file_name = os.path.basename(true_path)
        pred_path = os.path.join(pred_folder, file_name)
        
        if os.path.exists(pred_path):
            true_image = np.array(Image.open(true_path))
            pred_image = np.array(Image.open(pred_path))
            true_image = rgb_to_class_id(true_image, color_map)
            pred_image = rgb_to_class_id(pred_image, color_map)

            iou, pa, f1_score, f2_score = calculate_metrics_single_class(true_image, pred_image)
            iou_list.append(iou)
            pa_list.append(pa)
            f1_list.append(f1_score)
            f2_list.append(f2_score)

            tqdm.write(f"File: {file_name}, IOU: {iou:.4f}, PA: {pa:.4f}, F1: {f1_score:.4f}, F2: {f2_score:.4f}")

    visualize_metrics(iou_list, pa_list)
    print(f"Overall: Mean IOU = {np.nanmean(iou_list):.4f}, Mean PA = {np.nanmean(pa_list):.4f}, Mean F1 = {np.nanmean(f1_list):.4f}, Mean F2 = {np.nanmean(f2_list):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation metrics for single-class images.")
    parser.add_argument("--true_folder", type=str, required=True, help="Path to the folder containing ground truth images.")
    parser.add_argument("--pred_folder", type=str, required=True, help="Path to the folder containing predicted images.")
    
    args = parser.parse_args()
    
    process_folders(true_folder=args.true_folder, pred_folder=args.pred_folder)
