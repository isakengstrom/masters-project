import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from models.LSTM import LSTM
from train import train
from test import test
from dataset import FOIKineticPoseDataset

from helpers.paths import EXTR_PATH


if __name__ == "__main__":
    # Hyper parameters:
    hidden_size = 128
    num_classes = 10
    start_epoch = 1
    num_epochs = 2
    batch_size = 100
    learning_rate = 0.001

    input_size = 28
    sequence_len = 100
    num_layers = 2

    # Other params
    json_path = EXTR_PATH + "final_data_info.json"
    root_dir = EXTR_PATH + "final/"


    use_cuda = torch.cuda.is_available()

    kinetic_dataset = FOIKineticPoseDataset(json_path, root_dir, sequence_len)
    #train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=2)

    print("Dataset length: {}".format(len(kinetic_dataset)))
    for i in range(15000, 15010):
        item = kinetic_dataset[i]
        print(item["data"].shape)
    model = LSTM(input_size, hidden_size, num_layers, num_classes)

    if use_cuda:
        model.cuda()
        cudnn.benchmark = True

    objective = nn.TripletMarginLoss()

    optimizer = torch.optim.Adam(model.parameters())


    #model, loss_log, acc_log = train(model, train_loader, optimizer, objective, use_cuda, start_epoch, num_epochs)

