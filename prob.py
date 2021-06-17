import typing
import io
import os

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from urllib.request import urlretrieve

from PIL import Image
from torchvision import transforms

import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import os
import argparse
from apex import amp

from models import *
from utils import progress_bar
from flops import get_model_complexity_info
import logging
import math
from torch.optim.lr_scheduler import LambdaLR

imagenet_labels = dict(enumerate(open('label.txt')))

transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
im = Image.open("dog.jpg")
x = transform(im)
x.size()

model = ResNet0129()
checkpoint = torch.load("checkpoint/ckpt.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['net'])
model.eval()

outputs = model(x.unsqueeze(0))
probs = torch.nn.Softmax(dim=-1)(outputs)
top10 = torch.argsort(probs, descending=True)
print("Prediction Label!\n")
for idx in top10[0,]:
    print(f'{probs[0, idx.item()]:.5f} : {imagenet_labels[idx.item()]}')