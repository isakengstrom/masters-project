import os
import torch
import torch.nn as nn

def train(model, trainloader, optimizer, use_cuda, start_epoch, num_epochs):

