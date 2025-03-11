import torch
import torchvision
import torchvision.models as models
import cv2
import os
import gdown
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageOps


def check_and_download(file_path, gdrive_url):
    if os.path.exists(file_path):
        print(f"File '{file_path}' already exists.")
    else:
        print(f"File '{file_path}' not found. Downloading from Google Drive...")
        gdown.download(gdrive_url, file_path, quiet=False)
        if os.path.exists(file_path):
            print(f"Download complete: {file_path}")
        else:
            print("Download failed.")


def get_object_detection_model(num_classes):
    """Load Faster R-CNN model trained on COCO and modify for custom classes."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model


def apply_nms_and_conf_thresh(orig_prediction, iou_thresh=0.5, conf_thresh=0.5):
    """Apply Non-Maximum Suppression (NMS) and confidence threshold."""
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    keep = [int(i.cpu().numpy()) for i in keep if orig_prediction['scores'][i] >= conf_thresh]

    final_prediction = {
        'boxes': orig_prediction['boxes'][keep],
        'scores': orig_prediction['scores'][keep],
        'labels': orig_prediction['labels'][keep]
    }
    return final_prediction


def draw_boxes(image_tensor, boxes, scores, color=(0, 255, 0), thickness=2):
    """
    Draws bounding boxes and scores on an image tensor.

    Parameters:
    - image_tensor: torch.Tensor of shape (C, H, W), normalized in [0,1]
    - boxes: List of bounding boxes [(x_min, y_min, x_max, y_max), ...]
    - scores: List of scores corresponding to each box
    - color: Bounding box color (default: green)
    - thickness: Line thickness (default: 2)

    Returns:
    - image_with_boxes: np.ndarray of type int, shape (H, W, C), range [0, 255]
    """
    image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()

    # Convert grayscale to 3-channel if needed
    if image_np.shape[-1] == 1:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    # Draw bounding boxes and scores
    for box, score in zip(boxes, scores):
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # Compute text size relative to box size
        text = f"{score:.2f}"
        font_scale = max(0.5, (x_max - x_min) / 100.0)
        text_thickness = max(1, thickness // 2)
        
        # Put text
        cv2.putText(image_np, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, color, text_thickness, lineType=cv2.LINE_AA)
    
    return image_np  # (H, W, C), dtype=int


# Load model
num_classes = 2  # One class (class 0) is "background"
ckpt_path = './checkpoints/train/checkpoint.pth'
ckpt_link = "https://drive.google.com/uc?id=1V-yht4qfH4o5etdcR0QN-EaYZno4ee90"
model = get_object_detection_model(num_classes)
check_and_download(ckpt_path, ckpt_link)  
model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
model.eval()

# Streamlit UI
st.title("🔍 Mammogram Abnormalities Detection")
st.write("Upload mammogram images, and the model will detect abnormalities.")

# Sidebar for settings
st.sidebar.header("🔧 Settings")
iou_thresh = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.5, 0.01)
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

# File upload
uploaded_files = st.file_uploader("📤 Upload image(s)", type=["jpg", "png", "pgm", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    images = [Image.open(file) for file in uploaded_files]

    # Convert images to tensors
    transform = transforms.ToTensor()
    tensors = [transform(np.array(img)) for img in images]

    # Perform batch inference
    with torch.no_grad():
        predictions = model(tensors)

    # Apply NMS and thresholding
    filtered_predictions = [apply_nms_and_conf_thresh(pred, iou_thresh, conf_thresh) for pred in predictions]

    # Display results side-by-side
    for img, tensor, pred in zip(images, tensors, filtered_predictions):
        original = np.array(img)
        detected = draw_boxes(tensor, pred['boxes'], pred['scores'])

        col1, col2 = st.columns(2)
        with col1:
            st.image(original, caption="Original Image", use_container_width=True)
        with col2:
            st.image(detected, caption="Detected Abnormalities", use_container_width=True)
        
        # st.write(f"🔍 **Detected {len(pred['boxes'])} abnormalities**")
        # for i, (box, score) in enumerate(zip(pred['boxes'], pred['scores'])):
        #     st.write(f"📌 **Abnormality {i+1}**: Confidence {score:.2f} | Box {box.cpu().numpy().tolist()}")
