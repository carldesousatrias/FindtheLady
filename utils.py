# import nécessaire à tous
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision as tv
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# functions

def binomial(n, k):
    if not 0 <= k <= n:
        return 0
    b = 1
    for t in range(min(k, n-k)):
        b *= n
        b //= t+1
        n -= 1
    return b

def print_net(net):
    for name, parameters in net.named_parameters():
        print(name, ':', parameters.size())
    return

def inference(net, img, transform):
    """make the inference for one image and a given transform"""
    img_tensor= transform(img).unsqueeze(0)
    net.eval()
    with torch.no_grad():
        logits = net.forward(img_tensor.to(device))
        _, predicted = torch.max(logits, 1) # take the maximum value of the last layer
    return predicted

def fulltest(net,testloader):
    # test complet
    correct = 0
    total = 0

    # torch.no_grad do not train the network
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            if inputs.size()[1]   == 1:
                inputs.squeeze_(1)
                inputs = torch.stack([inputs, inputs, inputs], 1)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
    return 100 - (100 * float(correct) / total)

def calcul_diff(original,retrieved):
    ''' calculate the difference between the original and the retrieved order'''
    res=0
    for i in range(min(len(retrieved),len(original))):
        if original[i]!=retrieved[i]:res+=1
    return res

def dataloader(trainset,testset,batch_size=100):
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2)

    return trainloader,testloader

def CIFAR10_dataset():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # datasets
    trainset = tv.datasets.CIFAR10(
        root='./data/',
        train=True,
        download=True,
        transform=transform_train)

    testset = tv.datasets.CIFAR10(
        './data/',
        train=False,
        download=True,
        transform=transform_test)

    return trainset, testset, transform_test