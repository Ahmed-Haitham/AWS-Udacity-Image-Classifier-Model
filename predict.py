# Import Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms, models, datasets
from PIL import Image
from collections import OrderedDict
import argparse
import json
from workspace_utils import active_session

# Define Mandatory and Optional Arguments for the script
parser = argparse.ArgumentParser (description = "Prediction of Flower Classification")
parser.add_argument("--top_k", type = int, default = 5)
parser.add_argument("image_path", type = str)
parser.add_argument("input", type = str)
parser.add_argument("--category_names", type = str)
parser.add_argument("--gpu", type = str)
args = parser.parse_args()


# Loading the checkpoint and rebuild the model
def model(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained = True)
    if checkpoint ['arch'] == 'alexnet':
        model = models.alexnet(pretrained = True)
    else: #vgg16 as only 2 options available
        model = models.vgg16(pretrained = True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    #turning off tuning of the model
    for param in model.parameters():
        param.requires_grad = False
    
    return model

# Function to process a PIL image for use in a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    with Image.open(image) as img:
        # Get original image dimensions
        width, height = img.size
        
        #Find the shortest side and crop it to 256
        if width < height:
            size=[256, 256**600]
        else: 
            size=[256**600, 256]
            
        img.thumbnail(size)
        
        center = width/4, height/4
        left = center[0]-(244/2)
        top = center[1]-(244/2)
        right = center[0]+(244/2)
        bottom = center[1]+(244/2)
        crop = img.crop((left, top, right, bottom))
        
        np_image = np.array(crop)/255
        
        # Normalizing each color channel
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (np_image - mean) / std
        
        image = image.transpose((2, 0, 1)) 
    return image

# Defining prediction function
def predict(image_path, model, topk, device):
    img = process_image(image_path)
    img = torch.from_numpy(np.array([img])).float()
    
    model.to(device)
    img.to(device)
    
    with torch.no_grad():
        logps = model.forward(img)
        probability = torch.exp(logps)
        top_p, top_class = probability.topk(topk, dim = 1)
        
    # Detatch all of the details
    top_p = np.array(top_p.detach())[0] 
    top_class = np.array(top_class.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_class = [idx_to_class[lab] for lab in top_class]
        
    return top_p, top_class

# Setting values data loading
file_path = args.image_path

# Defining device: either cuda or cpu
device = "cuda" if args.gpu == 'GPU' else "cpu"
    

if args.category_names == True:
    with open(args.category_names, 'r') as F: cat_to_name = json.load(F)
else: 
    with open('ImageClassifier/cat_to_name.json', 'r') as F: cat_to_name = json.load(F) 
           

# Loading model from checkpoint provided
model = model(args.input)

num_classes = args.top_k if args.top_k == True else 5


top_p = predict(file_path, model, num_classes, device)[0]
top_class = predict(file_path, model, num_classes, device)[1]


for c in top_class:
    class_names = cat_to_name[c]
    
for n in range(num_classes):
    print("Number:", n+1, "/", num_classes)
    print("Class:", class_names[n])
    print("Probability:", top_p[n]*100, "%")
