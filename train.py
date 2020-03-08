# Filename : train.py
# Author   : Rahul Mannepally
# Course   : Udacity - The AI Programming with Python Nanodegree
# Date     : 15-09-2018

#!/usr/bin/env python3
# Train - Inputs tested with my code.
# python train.py ../aipnd-project/flowers --arch vgg --gpu
# python train.py ../aipnd-project/flowers --arch vgg --save_dir model --learning_rate 0.0003 --hidden_units 500 --epochs 4 --gpu


import sys
import argparse
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from torch.autograd import Variable
from torchvision import models, datasets, transforms

resnet34 = models.resnet34(pretrained=True)
vgg11    = models.vgg11(pretrained=True)

models   = {"resnet" : resnet34, "vgg" : vgg11}

#--------------------------------------------------------------------------------------------------------------------

''' Function to check if GPU/CUDA enabled 
'''
def check_gpu():
    gpu_status = torch.cuda.is_available() #is gpu available
    return gpu_status 

#---------------------------------------------------------------------------------------------------------------------

''' Function to select one of the four available models
'''
def fun_choosemodel(architecture):
    target_model = models[architecture]
    return target_model 
    
#---------------------------------------------------------------------------------------------------------------------

''' Function to evalute the loss and accuracy in a dataset 
'''
def fun_evalmodel(testloader, model, criterion, gpu):
    ''' Function is totally derived from Matt Leonard's PyTorch lessons
        Refer to Section 8 of PyTorch - Transfer learning. 
    '''
    accuracy = 0
    testloss = 0 
    
    if check_gpu() and gpu == True:
        model.cuda() 
        
    for inputs, labels in testloader: 
        inputs = Variable(inputs)
        labels = Variable(labels) 
            
        if check_gpu() and gpu == True:
            inputs = inputs.cuda()
            labels = labels.cuda() 
            
        output    = model.forward(inputs)
        testloss += criterion(output, labels).item()
        ps        = torch.exp(output) 
        equality  = (labels.data == ps.max(1)[1]) 
        accuracy += equality.type_as(torch.FloatTensor()).mean() 
        
    return (testloss/len(testloader)),(accuracy/len(testloader))

#----------------------------------------------------------------------------------------------------------------------------

''' Function to train a dataset and test a validation dataset against the trained dataset 
'''
def fun_trainmodel(model, trainloader, testloader, epochs, print_every, learning_rate, hidden_units, gpu, class_to_index, arch):   
    ''' Function is totally derived from Matt Leonard's PyTorch lessons
        Refer to Section 8 of PyTorch - Transfer learning. 
    '''
    class_to_index = class_to_index
    
    for param in model.parameters():
        param.required_grad = False     
   
    if arch == 'resnet':
        input_size = model.fc.in_features
    elif arch == 'vgg':
        input_size = model.classifier[0].in_features
    else:
        print("Unknown_model_error") 
              
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    if arch.startswith('vgg'):
        model.classifier = classifier
        params = model.classifier.parameters()
    elif arch.startswith('resnet'):
        model.fc = classifier
        params = model.fc.parameters()
    
    if check_gpu() and gpu == True:
        model.cuda() 
    
    steps      = 0 
    criterion  = nn.NLLLoss() 
    optimizer  = optim.Adam(params, lr = learning_rate)
    model.class_to_idx = class_to_index 
    model.train()
    
    for e in range(epochs):
        running_loss = 0
        
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1        
            inputs = Variable(inputs)
            labels = Variable(labels) 
            
            if check_gpu() and gpu == True:
                inputs = inputs.cuda()
                labels = labels.cuda() 
                
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss    = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step() 
            running_loss += loss.item() 
            
            if steps % print_every == 0: 
                model.eval() 
                eval_loss, eval_accuracy = fun_evalmodel(testloader, model, criterion, gpu)
                print("Epoch: {}/{}".format(e+1, epochs),
                      "Training Loss: {:.3f}".format(running_loss/print_every),
                      "Validation Loss: {:.3f}".format(eval_loss),
                      "Validation Accuracy: {:.3f}".format(eval_accuracy))
                running_loss = 0
                model.train() 
                
    return model, criterion, optimizer 

#---------------------------------------------------------------------------------------------------------------------

''' Function to parse command line input arguments and save them to necessary params 
'''
def get_input_arguments():
    args = {}
    parser = argparse.ArgumentParser(description = 'Image Classifier')
    parser.add_argument("image_dir")
    parser.add_argument('--arch', type=str, choices=["resnet", "vgg"], required=True)
    parser.add_argument('--learning_rate', type=float, default=0.0003)
    parser.add_argument('--hidden_units', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--save_dir', type=str, default="model")
    parsed_args = parser.parse_args()
    args = vars(parsed_args)
    return args

#---------------------------------------------------------------------------------------------------------------------

''' Function to load the data using torchvision 
'''
def get_data_loaders(source_dir):
    ''' Function is totally derived from PyTorch lessons
    '''
    train_dir = source_dir + '/train'
    valid_dir = source_dir + '/valid'
    test_dir  = source_dir + '/test'
    
    data_transforms = {
        'training': transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])]),
        
        'validation': transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])]),
        
        'testing': transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    }

    image_datasets = {
        'training'   : datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'validation' : datasets.ImageFolder(valid_dir, transform=data_transforms['validation']),
        'testing'    : datasets.ImageFolder(test_dir,  transform=data_transforms['testing'])
    }

    dataloaders = {
        'training'   : torch.utils.data.DataLoader(image_datasets['training'],   batch_size = 64, shuffle=True),
        'validation' : torch.utils.data.DataLoader(image_datasets['validation'], batch_size = 64, shuffle=True),
        'testing'    : torch.utils.data.DataLoader(image_datasets['testing'],    batch_size = 64, shuffle=False)
    }
    
    class_to_index = image_datasets['training'].class_to_idx
    
    return dataloaders, class_to_index

#---------------------------------------------------------------------------------------------------------------------

''' Function to select one of the four available models
'''
def save_model(model, trainloader, checkpoint_path, architecture, learning_rate, hidden_units, epochs, optimizer, class_to_index):
    ''' Function derived from PyTorch help section. 
        URL1: https://pytorch.org/docs/stable/torch.html
        URL2: https://pytorch.org/docs/stable/notes/serialization.html
    '''
    model.class_to_idx = class_to_index
    torch.save({
        'arch' : architecture,
        'learning_rate': learning_rate,
        'hidden_units' : hidden_units,
        'epochs' : epochs, 
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }, checkpoint_path + "/checkpoint.pth")

#---------------------------------------------------------------------------------------------------------------------

''' Main function for training model. 
'''    
def main():
    # Parsing input arguments from command line 
    args         = get_input_arguments()
    
    # Assigning source directory 
    source_dir   = args["image_dir"]
    
    # Get the dataloaders comprising of training, validation and test datasets 
    dataloaders, class_to_index = get_data_loaders(source_dir) 
    
    # Choose architecture and create the model 
    model = fun_choosemodel(args["arch"])
    
    if args["gpu"] == True and check_gpu():
        model.cuda() 
        print("CUDA is available!")
    
    # Train the model and test it against the validation data 
    model, criterion, optimizer = fun_trainmodel(
        model,
        dataloaders['training'],
        dataloaders['validation'],
        args["epochs"],
        40,
        args["learning_rate"],
        args["hidden_units"],
        args["gpu"],
        class_to_index,
        args["arch"]
    )
    
    # Save the model to check point for future use 
    save_model(model, dataloaders['training'], args["save_dir"], args["arch"], args["learning_rate"], args["hidden_units"], args["epochs"], optimizer, class_to_index)
    
    # Call eval model to determine the loss and accuracy of the training data versus validation & test data 
    loss, accuracy = fun_evalmodel(dataloaders['validation'], model, criterion, args["gpu"])
    print("Validation Data Loss {:.3f}".format(loss), "Validation Data Accuracy: {:.3f}".format(accuracy))
    loss, accuracy = fun_evalmodel(dataloaders['testing'], model, criterion, args["gpu"])
    print("Testing Data Loss {:.3f}".format(loss), "Testing Data Accuracy: {:.3f}".format(accuracy))
    
if __name__ == "__main__":
    main()