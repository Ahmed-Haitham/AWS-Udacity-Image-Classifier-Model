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
parser = argparse.ArgumentParser(description = "Training Model for Flower Classification")
parser.add_argument("data_dir", type = str)
parser.add_argument("--arch", type = str, default = "vgg16")
parser.add_argument("--save_dir", type = str)
parser.add_argument("--learn_rate", type = float, default = 0.003)
parser.add_argument("--hidden_units", type = int, default = 512)
parser.add_argument("--epochs", type = int, default = 1)
parser.add_argument("--gpu", type = str)
args = parser.parse_args()


# Setting values data loading
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# Define processing device: cuda or cpu
device = "cuda" if args.gpu == 'GPU' else "cpu"


# Data loading
if data_dir:
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)
    
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size = 64)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size = 64)
    # End of data loading
    
# Load in a mapping from category label to category name
with open('ImageClassifier/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# Define the model
def model(arch, hidden_units): 
    if arch == "vgg16":
        model = models.vgg16(pretrained = True)
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        if hidden_units == False:
            model.classifier = nn.Sequential(OrderedDict([
                                  ("fc1", nn.Linear(25088, 4096)),
                                  ("relu1", nn.ReLU()),
                                  ("fc2", nn.Linear(4096, 1024)),
                                  ("relu2", nn.ReLU()),
                                  ("fc3", nn.Linear(1024, 512)),
                                  ("relu3", nn.ReLU()),
                                  ("dropout", nn.Dropout(p = 0.2)),
                                  ("fc4", nn.Linear(512, 102)),
                                  ("output", nn.LogSoftmax(dim = 1))]))
            
        else:
            model.classifier = nn.Sequential(OrderedDict([
                                  ("fc1", nn.Linear(25088, 4096)),
                                  ("relu1", nn.ReLU()),
                                  ("fc2", nn.Linear(4096, 1024)),
                                  ("relu2", nn.ReLU()),
                                  ("fc3", nn.Linear(1024, hidden_units)),
                                  ("relu3", nn.ReLU()),
                                  ("dropout", nn.Dropout(p = 0.2)),
                                  ("fc4", nn.Linear(hidden_units, 102)),
                                  ("output", nn.LogSoftmax(dim = 1))]))
            
    else:
        arch = "alexnet"
        model = models.alexnet(pretrained = True)
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
        
        if hidden_units == False:
            model.classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(9216, 4096)),
                                ('relu1', nn.ReLU()),
                                ('fc2', nn.Linear(4096, 1024)),
                                ('relu2', nn.ReLU()),
                                ('fc3', nn.Linear(1024, 512)),
                                ('relu3', nn.ReLU()),
                                ('dropout', nn.Dropout(p = 0.2)),
                                ('fc4', nn.Linear(512, 102)),
                                ('output', nn.LogSoftmax(dim = 1))]))
            
        else:
            model.classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(9216, 4096)),
                                ('relu1', nn.ReLU()),
                                ('fc2', nn.Linear(4096, 1024)),
                                ('relu2', nn.ReLU()),
                                ('fc3', nn.Linear(1024, hidden_units)),
                                ('relu3', nn.ReLU()),
                                ('dropout', nn.Dropout(p = 0.2)),
                                ('fc4', nn.Linear(hidden_units, 102)),
                                ('output', nn.LogSoftmax(dim = 1))]))
    
    return arch, model
            

# Define the validation function
def validation (model, validloader, criterion):
    # start our validation
    model.to(device)
    valid_loss = 0
    accuracy = 0
    
    for inputs, labels in validloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
        valid_loss += batch_loss.item()
        
        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim = 1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    return valid_loss, accuracy


arch = model(args.arch, args.hidden_units)[0]
model = model(args.arch, args.hidden_units)[1]

# Starting the training of the model
# Set up criterion
criterion = nn.NLLLoss()

# Set up optimizer 
optimizer = optim.Adam(model.classifier.parameters(), lr = args.learn_rate) if args.learn_rate == True else optim.Adam(model.classifier.parameters(), lr = 0.003)

model.to(device)    

# Setting number of epochs
epochs = args.epochs if args.epochs == True else 1

with active_session():
    # Let's use the model
    steps = 0
    train_loss = 0
    print_every = 5
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() # Optimizer is working on classifier paramters only
            
            # Forward and backward passes
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if steps % print_every == 0:
                # start our validation
                model.eval() # Switching to evaluation mode so that dropout is turned off

                with torch.no_grad(): # Turn off gradients for validation, saves memory and computations
                    valid_loss, accuracy = validation(model, validloader, criterion)
                    
                print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_loss/print_every),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                  "Valid Accuracy: {:.3f}%".format(accuracy/len(validloader)*100))

            train_loss = 0     
            model.train() # Make sure training is back on   

            
# Saving the model (checkpoint)
model.to('cpu') # No need to use cuda for saving/loading model

checkpoint = {'arch' : arch,
              'input_size': 25088,
              'output_size': 102,
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'class_to_idx': train_datasets.class_to_idx,
              'epochs': 10,
              'dropout': 0.2,
              'optimizer_state': optimizer.state_dict}


torch.save(checkpoint, args.save_dir + '/checkpoint.pth') if args.save_dir == True else torch.save(checkpoint, 'checkpoint.pth')

    