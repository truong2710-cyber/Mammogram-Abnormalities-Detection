import numpy as np # linear algebra
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageOps
import torchvision.models as models

def get_object_detection_model(num_classes):
  # load a model pre-trained on COCO
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  # get number of input features for the classifier
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  # replace the pre-trained head with a new one
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
  return model

def apply_nms_and_conf_thresh(orig_prediction, iou_thresh=0.01, conf_thresh=0.5):
  # torchvision returns the indices of the bboxes to keep
  keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
  keep = [int(i.cpu().numpy()) for i in keep if orig_prediction['scores'][i] >=conf_thresh]
  final_prediction = orig_prediction.copy()
  final_prediction['boxes'] = final_prediction['boxes'][keep]
  final_prediction['scores'] = final_prediction['scores'][keep]
  final_prediction['labels'] = final_prediction['labels'][keep]
  
  return final_prediction

def plot_img_bbox(img, target):
  # plot the image and bboxes
  # Bounding boxes are defined as follows: x-min y-min width height
  fig, a = plt.subplots(1,1)
  fig.set_size_inches(10,10)
  if img.shape[0] == 3:
    img = img.permute(1,2,0)
  a.imshow(img)
  for box in (target['boxes']):
    x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
    rect = patches.Rectangle(
      (x, y),
      width, height,
      linewidth = 1,
      edgecolor = 'r',
      facecolor = 'none'
    )
    # Draw the bounding box on top of the image
    a.add_patch(rect)
  plt.axis('off')
  return fig
    
num_classes = 2 # one class (class 0) is dedicated to the "background"

# get the model using our helper function
model = get_object_detection_model(num_classes)
model.load_state_dict(torch.load('./checkpoints/train/checkpoint.pth',map_location=torch.device('cpu')))
import streamlit as st
st.write("""
         # Mammogram Abnormalities Detection
         """
         )
st.write("This is a simple mammogram abnormalities detection web app")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    img = image.save("img.jpg")
    img = cv2.imread("img.jpg")
    transform = transforms.ToTensor()

    # Convert the image to PyTorch tensor
    tensor = transform(img)
    st.write("Prediction:")
    model.eval()
    with torch.no_grad():
        prediction = model([tensor])[0]
        prediction = apply_nms_and_conf_thresh(prediction)
        fig = plot_img_bbox(tensor, prediction)
        #plt.imsave("fig.png",fig)
        st.pyplot(fig)
        #print(prediction)