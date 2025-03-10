import torch
import torchvision
import torchvision.models as models
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageOps


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


def draw_boxes(image_tensor, boxes, color=(0, 255, 0), thickness=2):
    """
    Draws bounding boxes on an image tensor.

    Parameters:
    - image_tensor: torch.Tensor of shape (C, H, W), normalized in [0,1]
    - boxes: List of bounding boxes [(x_min, y_min, x_max, y_max), ...]
    - color: Bounding box color (default: green)
    - thickness: Line thickness (default: 2)

    Returns:
    - image_with_boxes: np.ndarray of type int, shape (H, W, C), range [0, 255]
    """
    image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()

    # Convert grayscale to 3-channel if needed
    if image_np.shape[-1] == 1:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    # Draw bounding boxes
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), color, thickness)

    return image_np  # (H, W, C), dtype=int


# Load model
num_classes = 2  # One class (class 0) is "background"
model = get_object_detection_model(num_classes)
model.load_state_dict(torch.load('./checkpoints/train/checkpoint.pth', map_location=torch.device('cpu')))
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
        detected = draw_boxes(tensor, pred['boxes'])

        col1, col2 = st.columns(2)
        with col1:
            st.image(original, caption="Original Image", use_container_width=True)
        with col2:
            st.image(detected, caption="Detected Abnormalities", use_container_width=True)
        
        # st.write(f"🔍 **Detected {len(pred['boxes'])} abnormalities**")
        # for i, (box, score) in enumerate(zip(pred['boxes'], pred['scores'])):
        #     st.write(f"📌 **Abnormality {i+1}**: Confidence {score:.2f} | Box {box.cpu().numpy().tolist()}")
