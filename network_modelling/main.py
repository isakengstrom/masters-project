import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np

from models.LSTM import LSTM
from train import train
from test import test
from dataset import FOIKineticPoseDataset as FOID
from helpers.paths import EXTR_PATH
from transforms import FilterJoints, NormalisePose, ChangePoseOrigin, ToTensor, NormalisePoses

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

    # Pose transforms
    filter_pose = FilterJoints()
    normalise_pose = NormalisePose()
    change_pose_origin = ChangePoseOrigin()

    # Transforms
    to_tensor = ToTensor()
    composed = transforms.Compose([FilterJoints(), NormalisePoses(low=2, high=5), ToTensor()])

    foid = FOID(json_path, root_dir, sequence_len, transform=composed, pose_transform=None)

    foid_item = foid[0]
    seq = foid_item["sequence"]
    shape = ""

    if isinstance(seq, np.ndarray):
        shape = seq.shape
    elif isinstance(seq, list):
        shape = len(seq)
    elif isinstance(seq, torch.Tensor):
        shape = seq.size()

    print("Dataset instance with id '{}' is of type '{}', with shape {}"
          .format(foid_item["id"], type(seq), shape))

    model = LSTM(input_size, hidden_size, num_layers, num_classes)

    if use_cuda:
        model.cuda()
        cudnn.benchmark = True

    objective = nn.TripletMarginLoss()

    optimizer = torch.optim.Adam(model.parameters())


    #model, loss_log, acc_log = train(model, train_loader, optimizer, objective, use_cuda, start_epoch, num_epochs)

