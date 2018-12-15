"""Package Imports"""
# importing needed packeges
import matplotlib.image as mpimg

import matplotlib.pyplot as plt

from collections import OrderedDict

import torch
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
import os, sys

from torch.autograd import Variable

import argparse
""" """


parser = argparse.ArgumentParser(description='Image Classifier - Training Part')
parser.add_argument('--data_dir', type=str, action="store", default="flowers", help='the directory of flower images')
parser.add_argument('--gpu', dest='gpu', action='store_true', default="gpu", help='activate the GPU during the training')
parser.add_argument('--save_dir', type=str,dest="save_dir", action="store", default="checkpoint.pth", help='directory to save checkpoints')
parser.add_argument('--arch', dest='arch', action="store", default="vgg16", type = str, help='model architecture')
parser.add_argument('--learning_rate', type=float, dest="learning_rate", action="store", default=0.001, help='learning rate')
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=9000, help='number of hidden units')
parser.add_argument('--epochs', type=int, dest="epochs", action="store", default=5, help='number of epochs')
parser.add_argument('--dropout', type=float, dest = "dropout", action = "store", default = 0.5, help='dropout percentage')

arg_parser = parser.parse_args()

data_dir = arg_parser.data_dir
gpu = arg_parser.gpu
save_dir = arg_parser.save_dir
model_arch = arg_parser.arch
lr = arg_parser.learning_rate
hidden_units = arg_parser.hidden_units
epochs = arg_parser.epochs
dropout = arg_parser.dropout

image_datasets_train = None

def load_data():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    """Training data augmentation + Data normalization"""
    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    """Data loading"""
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    """Data batching"""
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    return train_data, valid_data, test_data

def model_architecture(lr=0.001, hidden_units=9000):
    # TODO: Build and train your network

    if model_arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif model_arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_arch == 'vgg11':
        model = models.vgg11(pretrained=True)
    else:
        print("Please choose either densenet121, VGG11 or VGG16")

    # Hyperparameters for our network
    input_size = 1024
    hidden_sizes = [500, 500]
    output_size = 102

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_sizes[0])), # in_features will be different depending on the network you choosed
                              ('relu', nn.ReLU()),
                              ('drpot', nn.Dropout(p=dropout)),
                              ('fc2', nn.Linear(hidden_sizes[1], output_size)), # out_features = 2, because it's 2 classes , in the project it's different
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    
    model.classifier = classifier

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return model, criterion, optimizer

# Implement a function for the validation pass
def validation(model, validloader, criterion):
    
    model.eval()
    valid_loss = 0
    accuracy = 0
    for images, labels in validloader:

        ### code reference ###
        # https://github.com/bryanfree66/AIPND_image_classification/blob/master/Image%20Classifier%20Project.ipynb
        images = Variable(images.float().cuda(), volatile=True)
        labels = Variable(labels.long().cuda(), volatile=True)
        #####################

        output = model.forward(images) #%%
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy

def do_deep_learning(epochs=5):
    
    # hyperparameters
    epochs = 10
    steps = 0
    print_every = 40

    # criterion & optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    steps = 0

    if gpu == 'gpu':
        model.to('cuda')
    else:
        print("the model will be trained using gpu due to the performance")
        model.to('cuda')

    for e in range(epochs):
        running_loss = 0
        model.train() #
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()
                
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion)
                
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                
                running_loss = 0
                
                # Make sure training is back on
                model.train()

def save_checkpoint(model):
    # TODO: Save the checkpoint
    model.class_to_idx = image_datasets_train.class_to_idx

    checkpoint = {'arch': model_arch,
                  'state_dict': model.state_dict(),
                  'optimizer_dict': optimizer.state_dict(),
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'clssifier': model.classifier,
                  'learning_rate': lr}

    torch.save(checkpoint, save_dir)

if __name__== "__main__":

    print ("start training ...")
    image_datasets_train, image_datasets_valid, image_datasets_test = load_data()
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(image_datasets_train, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(image_datasets_valid, batch_size=64)
    testloader = torch.utils.data.DataLoader(image_datasets_test, batch_size=64)
    model, criterion, optimizer = model_architecture(lr, hidden_units)
    do_deep_learning(epochs)
    save_checkpoint(model)
    print ("end training ...")
