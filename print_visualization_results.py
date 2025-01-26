import cv2
import numpy as np
import torch
import os
import random
from PIL import Image
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config

# Setup configuration and model
def setup_model():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    # Adjust the paths to your configuration and model weights
    cfg.merge_from_file("/home/gianluca/PycharmProjects/benchmark_loss/Mask2Former/configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_tiny_bs16_90k_2DEC.yaml")
    # cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl'
    cfg.MODEL.WEIGHTS = '/home/gianluca/PycharmProjects/benchmark_loss/Mask2Former/output/model_final_kan.pth'
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    return cfg, DefaultPredictor(cfg)

cfg, predictor = setup_model()

def predict(image, cfg):
    image = np.array(image)[:, :, ::-1]  # Convert PIL image to OpenCV format (BGR)
    outputs = predictor(image)

    # Visualize results
    if len(cfg.DATASETS.TEST):
        dataset_name = cfg.DATASETS.TEST[0]
    else:
        dataset_name = "__unused"
    metadata = MetadataCatalog.get(dataset_name)
    v = Visualizer(image, metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
    return Image.fromarray(semantic_result[:, :, ::-1])  # Convert back to RGB

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run semantic segmentation on images in a directory containing subdirectories.')
    parser.add_argument('--input_dir', required=True, help='Path to the input directory containing subdirectories with images.')
    parser.add_argument('--output_dir', required=True, help='Path to the output directory where results will be saved.')
    parser.add_argument('--num_images', type=int, default=20, help='Total number of images to process.')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of subdirectories in the input directory
    subdirs = [os.path.join(args.input_dir, d) for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]

    if not subdirs:
        print("No subdirectories found in the input directory.")
    else:
        processed_images = 0
        attempts = 0  # To prevent infinite loops in case of insufficient images
        max_attempts = args.num_images * 10  # Arbitrary large number

        while processed_images < args.num_images and attempts < max_attempts:
            attempts += 1
            # Randomly select a subdirectory
            subdir = random.choice(subdirs)
            # Get list of images in the selected subdirectory
            image_files = [f for f in os.listdir(subdir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            if not image_files:
                continue  # Skip if no images in subdirectory

            # Randomly select an image from the subdirectory
            image_name = random.choice(image_files)
            image_path = os.path.join(subdir, image_name)
            try:
                image = Image.open(image_path).convert("RGB")
                result_image = predict(image, cfg)
                # Save the result
                output_filename = f"result_{processed_images}_{os.path.basename(subdir)}_{image_name}"
                output_path = os.path.join(args.output_dir, output_filename)
                result_image.save(output_path)
                print(f"Processed {image_name} from {os.path.basename(subdir)} and saved result to {output_path}")
                processed_images += 1
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        if processed_images == 0:
            print("No images processed. Please check the input directory and ensure it contains subdirectories with images.")
        elif processed_images < args.num_images:
            print(f"Only {processed_images} images were processed due to limited images available.")

