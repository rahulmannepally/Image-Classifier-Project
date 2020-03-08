# Filename : predict.py
# Author   : Rahul Mannepally
# Course   : Udacity - The AI Programming With Python Nanodegree
# Date     : 16-09-2018

#!/usr/bin/env python3
# Predict - Input tested for my code
# python predict.py ../aipnd-project/flowers/test/28/image_05230.jpg model/checkpoint.pth --gpu
# python predict.py ../aipnd-project/flowers/test/28/image_05230.jpg model/checkpoint.pth --topk 5 --gpu
# python predict.py ../aipnd-project/flowers/test/28/image_05230.jpg model/checkpoint.pth --topk 5 --category_names ../aipnd-project/cat_to_name.json --gpu

import argparse
import json
import torch
import numpy as np
from PIL import Image
import torchvision.models as models
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F


resnet34 = models.resnet34(pretrained=True)
vgg11    = models.vgg11(pretrained=True)

models   = {"resnet" : resnet34, "vgg" : vgg11}

#-------------------------------------------------------------------------------------------------------------------
''' Function to check if GPU/CUDA enabled 
'''
def check_gpu():
    gpu_status = torch.cuda.is_available()
    return gpu_status 

#-------------------------------------------------------------------------------------------------------------------
''' Function to select one of the four available models
'''
def choose_model(architecture):
    target_model = models[architecture]
    return target_model 

#---------------------------------------------------------------------------------------------------------------------
''' Function to get input arguments from command line 
'''
def get_input_arguments():
    args = {}
    parser = argparse.ArgumentParser(description = "Image Predictor") 
    # Mandatory items "input" and "checkpoint" don't have a default option
    parser.add_argument("input")
    parser.add_argument("checkpoint")
    # Optional items are assigned with default options 
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--category_names', type=str)
    parser.add_argument('--gpu', action='store_true')
    parsed_arguments = parser.parse_args() 
    args = vars(parsed_arguments)
    return args

#----------------------------------------------------------------------------------------------------------------------
''' Function to load model from checkpoint
'''
def load_model(checkpoint_path, gpu):
    # Re-load checkpoint from our training exercise 
    checkpoint   = torch.load(checkpoint_path)
    
    # Re-load architecture and create a model 
    architecture = checkpoint["arch"]
    model = choose_model(architecture) 
    
    # Hidden units is needed to create the classifier. 
    hidden_units = checkpoint['hidden_units']
    
    # Choose input size based on the model used for training 
    if architecture == 'resnet':
        input_size = model.fc.in_features
    elif architecture == 'vgg':
        input_size = model.classifier[0].in_features
    else:
        print("Unknown_model_error") 
     
    # Create a classifier for the give input size and hidden units 
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    # model.classifier = classifier
    
    if architecture.startswith('vgg'):
        model.classifier = classifier
    elif architecture.startswith('resnet'):
        model.fc = classifier    
    
    if check_gpu() and gpu == True:
        model.cuda() 
        
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    
    # Print debug information just to confirm what got re-loaded 
    print("Loaded from checkpoint path => {} with architecture: {}, hidden units: {}, epochs: {}".format
          (checkpoint_path, 
           checkpoint['arch'],
           checkpoint['hidden_units'],
           checkpoint['epochs']))
    
    return model

#---------------------------------------------------------------------------------------------------------------------
''' Function to translate category to name
'''
def translate_cat_to_name(path):
    # Dummy list 
    cat_to_name = []
    
    # Open the json file and extract & store class names to the list 
    with open(path, 'r') as fp:
        cat_to_name = json.load(fp)
        
    return cat_to_name 

#---------------------------------------------------------------------------------------------------------------------
''' Function to process the image
'''
def process_image(image):  
    # Get the current image size 
    width, height = image.size 
    tgt_size      = 256
    
    # To re-size we don't know which one is the shortest side, so check for height vs width 
    # and then decide to resize appropriately 
    if height > width:
        width = int(tgt_size)
        height = int(max(height * tgt_size/width, 1)) 
    else:
        height = int(tgt_size)
        width = int(max(width * tgt_size/height, 1)) 
    
    # Perform image resize 
    resized_img = image.resize((width, height)) 
    
    # Prep for image crop 
    tgt_size = 224
    width, height = resized_img.size 
    x1 = (width - tgt_size) / 2
    x2 = (height - tgt_size) / 2
    x3 = x1 + tgt_size
    x4 = x2 + tgt_size
    
    # Crop the image 
    crop_img = resized_img.crop((x1, x2, x3, x4))
    
    # Convert the image to Numpy array of float data and process it
    np_image = np.array(crop_img)/255. 
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])     
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    return np_image

#----------------------------------------------------------------------------------------------------------------------
''' Function to predict the image
'''
def predict_image(image_path, model, topk=5):
    model.eval()
    
    if check_gpu(): 
        model.cuda()
    
    # Open the image 
    image = Image.open(image_path)
    
    # Resize and crop the image to necessary size 
    np_image = process_image(image) 
    
    if check_gpu():
        input = torch.FloatTensor(np_image).cuda()
    else:
        input = torch.FloatTensor(np_image)
    
    # Detect the image-class and determine their probability 
    input.unsqueeze_(0)
    output = model(input)
    ps = F.softmax(output, dim = 1)
    probs, classes = torch.topk(ps, topk)
    
    # Store the inverted list and get the top probability items. 
    inverted_class_2_index = {model.class_to_idx[x]: x for x in model.class_to_idx}    
    new_classes = list() 
    
    for index in classes.cpu().numpy()[0]:
        new_classes.append(inverted_class_2_index[index])
        
    return probs.cpu().detach().numpy()[0], new_classes        

#---------------------------------------------------------------------------------------------------------------------
''' Main funtion for Image Predictor 
'''
def main():
    # Parse the input arguments from command line 
    arguments = get_input_arguments()
    
    # Create the model from checkpoint 
    model = load_model(arguments["checkpoint"], arguments["gpu"])
    
    # Use GPU if available 
    if arguments["gpu"] == True and check_gpu():
        model.cuda() 
        print("CUDA is available!")
    
    # Get the predicted image classes and their probability 
    probs, classes = predict_image(arguments["input"], model, arguments["topk"])   
    
    # If no path is provided for json file, then print just the classes and probability of top most item 
    if arguments["category_names"] == None: 
        for item in range(arguments["topk"]):
            print("Class of Flower: {} & Probability: {:.3f}".format(classes[item], probs[item]))
    # else the json path is valid, and we print the top "k" items with their class names and probability 
    else:
        cat_to_name = translate_cat_to_name(arguments["category_names"])
        for item in range(arguments["topk"]):
            print("Class of Flower: {} & Probability: {:.3f}".format(cat_to_name[classes[item]], probs[item]))
    
if __name__ == "__main__":
    main()