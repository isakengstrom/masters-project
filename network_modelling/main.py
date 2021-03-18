import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from models.Net import LSTM
from train import train
from test import test




if __name__ == "__main__":
    # Hyper parameters:
    hidden_size = 128
    num_classes = 10,
    num_epochs = 2
    batch_size = 100
    learning_rate = 0.001

    input_size = 28
    sequence_length = 28
    num_layers = 2

    model = LSTM(input_size, hidden_size, num_layers, num_classes)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
        cudnn.benchmark = True


