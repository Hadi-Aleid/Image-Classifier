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
import json
""" """



parser = argparse.ArgumentParser(description='Image Classifier - Prediction Part')

parser.add_argument('--input', default='./flowers/test/1/image_06752.jpg', action="store", type = str, help='image path')
#parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/", help='')
parser.add_argument('--checkpoint', default='./checkpoint.pth', action="store", type = str, help='checkpoint file path/name')
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int, help='return top K most likely classes')
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json', help='mapping the categories to real names')
parser.add_argument('--gpu', dest='gpu', action='store_true', default="gpu", help='activate the GPU during the prediction')

arg_parser = parser.parse_args()

image_path = arg_parser.input
model_path = arg_parser.checkpoint
topk = arg_parser.top_k
category_names = arg_parser.category_names
gpu = arg_parser.gpu


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    learning_rate = checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    
    if model_arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif model_arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_arch == 'vgg11':
        model = models.vgg11(pretrained=True)
    else:
        print("Please choose either densenet121, VGG11 or VGG16")
        model = models.vgg16(pretrained=True)

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
                              ('drpot', nn.Dropout(p=0.5)),
                              ('fc2', nn.Linear(hidden_sizes[1], output_size)), # out_features = 2, because it's 2 classes , in the project it's different
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    
    model.classifier = classifier
    
    if gpu == 'gpu':
        model.to('cuda')
    else:
        print("the model will be trained using gpu due to the performance")
        model.to('cuda')
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer_dict = checkpoint['optimizer_dict']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    size = 256
    width, height = pil_image.size

    shortest_side = min(width, height)

    pil_image = pil_image.resize((int((pil_image.width/shortest_side)*size), int((pil_image.height/shortest_side)*size)))

    img_loader = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    # pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()

    np_image = np.array(pil_image)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return np_image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    
    if gpu == 'gpu':
        model.to('cuda')
    else:
        print("the model will be trained using gpu due to the performance")
        model.to('cuda')
    

    # Image Preprocessing
    np_array = process_image(image_path)
    image = torch.from_numpy(np_array)
    image = Variable(image.float().cuda(), volatile = True)
    image = image.unsqueeze(0)
    output = model.forward(image)

    ps = torch.exp(output)

    return ps.topk(topk)

def Sanity_Checking(probs, classes):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    probs, classes = predict('flowers/test/10/image_07104.jpg', model)

    # start  mapping part
    probs = probs.data.cpu().numpy()[0]
    indexes = classes.data.cpu().numpy()[0]
    print ('probs: ', probs, 'indexes: ',indexes)

    # Convert indices to classes
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_indexes = [index_to_class[each] for each in indexes]

    print ('x_prob: ', probs,'top_indexes: ', top_indexes)

    classes = []
    for x in top_indexes:
        classes.append(cat_to_name[x])

    print ('flower_class: ',classes)


if __name__== "__main__":

    print ("start Prediction ...")
    model = load_checkpoint(model_path)
    probs, classes = predict(image_path, model, topk)
    Sanity_Checking(probs, classes)
    print ("end Prediction ...")
