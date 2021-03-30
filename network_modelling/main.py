import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from models.LSTM import LSTM
from train import train
from test import test
#from dataset import FOIKineticPoseDataset, NormalisePose, FilterPose, Pose # , ChangePoseOrigin
from dataset2 import FOIKineticPoseDataset as FOID
from helpers.paths import EXTR_PATH
from transforms import FilterPose, NormalisePose, ToTensor

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

    filter_pose = FilterPose()
    normalise_pose = NormalisePose()
    #change_pose_origin = ChangePoseOrigin()

    pose_composed = transforms.Compose([FilterPose(), NormalisePose(low=0, high=1)])

    to_tensor = ToTensor()
    #composed = transforms.Compose([normalise, change_pose_origin])

    foid = FOID(json_path, root_dir, sequence_len, transform=to_tensor, pose_transform=pose_composed)

    foid_item = foid[0]
    #print(len(foid_item["sequence"]))
    print(len(foid_item["sequence"][0]))
    '''
    for idx in range(0,10):
        foid_item = foid[idx]
        print(len(foid_item["sequence"]))
        print(foid_item["sequence"][0])
    '''


    '''
    kinetic_dataset = FOIKineticPoseDataset(json_path, root_dir, sequence_len, pose_transform=filter_poses)
    #train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=2)

    
    print("Dataset length: {}".format(len(kinetic_dataset)))
    item = kinetic_dataset[0]
    print(item["poses"][0])
    pose = Pose(item["poses"][0])
    print(pose.joints[23].name)
    '''



    #filter_poses(item["keypoints"][0])
    #print(item["poses"].shape)

    '''
    for i in range(15000, 15010):
        item = kinetic_dataset[i]
        print(item["keypoints"].shape)
    '''
    model = LSTM(input_size, hidden_size, num_layers, num_classes)

    if use_cuda:
        model.cuda()
        cudnn.benchmark = True

    objective = nn.TripletMarginLoss()

    optimizer = torch.optim.Adam(model.parameters())


    #model, loss_log, acc_log = train(model, train_loader, optimizer, objective, use_cuda, start_epoch, num_epochs)

