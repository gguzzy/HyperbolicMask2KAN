import gradio as gr
import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from mask2former import add_maskformer2_config
from PIL import Image
from detectron2.projects.deeplab import add_deeplab_config

# Setup configuration and model
def setup_model():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    # cfg.merge_from_file("/home/gianluca/PycharmProjects/benchmark_loss/Mask2Former/configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_tiny_bs16_90k_2DEC.yaml")
    cfg.merge_from_file("/home/gianluca/PycharmProjects/benchmark_loss/Mask2Former/configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_tiny_bs16_90k_2DEC.yaml")
    # cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl'
    cfg.MODEL.WEIGHTS = '/home/gianluca/PycharmProjects/benchmark_loss/Mask2Former/output/model_final_kan.pth'
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    #cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    #cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
    return DefaultPredictor(cfg)


predictor = setup_model()


def predict(image):
    image = np.array(image)[:, :, ::-1]  # Convert PIL image to OpenCV format
    outputs = predictor(image)

    # Visualize results
    coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")

    # # Panoptic segmentation
    # v = Visualizer(image, coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    # panoptic_result = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"), outputs["panoptic_seg"][1]).get_image()
    #
    # # Instance segmentationS
    # v = Visualizer(image, coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    # instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()

    # Semantic segmentation
    v = Visualizer(image, coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()

    # Concatenate results
    # result_image = np.concatenate((panoptic_result, instance_result, semantic_result), axis=0)[:, :, ::-1]

    return Image.fromarray(semantic_result) # Image.fromarray(result_image)


# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),  # Updated input type
    outputs=gr.Image(type="pil"),  # Updated output type
    title="Mask2Kan Semantic Segmentation Demo",
    description="Upload an image to see semantic segmentation results."
)

if __name__ == "__main__":
    iface.launch()
