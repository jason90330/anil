'''
cross entropy, contrastive 前有加 Normalize，Log 較詳細，但分數不是最高
'''
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
# from _typeshed import NoneType
import os
import random
import argparse
from pytorch_metric_learning import reducers
from pytorch_metric_learning import distances
import wandb
import gc

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision as tv

import learn2learn as l2l
from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels
from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.reducers import MultipleReducers, ThresholdReducer, MeanReducer, ClassWeightedReducer, DoNothingReducer
from pytorch_metric_learning.distances import BaseDistance
from efficientnet_pytorch import EfficientNet
from dataset.customDataAnilAntiSupConv import customData
from torchvision import transforms
from misc.test import TestSupConv_v2

from statistics import mean
from copy import deepcopy


class Lambda(nn.Module):

    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)

class FeatExtr(nn.Module):
    """backbone + projection head"""
    def __init__(self, feat_dim=128):
        super(FeatExtr, self).__init__()
        self.encoder = EfficientNet.from_pretrained("efficientnet-b0", advprop=True)
        self.out_feature = self.encoder._fc.in_features
        ''' change here
        self.projHead =  torch.nn.Sequential(
                                        nn.AdaptiveAvgPool2d(1), 
                                        nn.Flatten(start_dim=1), 
                                        nn.Dropout(0.2),
                                        nn.Linear(out_feature, out_feature),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(out_feature, feat_dim))
        '''
        self.projHead =  torch.nn.Sequential(
                                        nn.AdaptiveAvgPool2d(1), 
                                        nn.Flatten(start_dim=1), 
                                        nn.Dropout(0.2))
       
    def forward(self, x):
        feat = self.encoder.extract_features(x)
        feat = self.projHead(feat)
        '''
        feat = F.normalize(self.projHead(feat), dim=1)
        '''
        return feat
    

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def log(trainErr, trainSupConvErr, trainAcc, valErr, valSupConvErr, valAcc, testErr, testSupConvErr, testAcc):
    wandb.log({"Meta Train Error": trainErr,
               "Meta Train SupConv Error": trainSupConvErr,
               "Meta Train Accuracy": trainAcc,
               "Meta Valid Error": valErr,
               "Meta Valid SupConv Error": valSupConvErr,
               "Meta Valid Accuracy": valAcc,
               "Meta Test Error": testErr,
               "Meta Test SupConv Error": testSupConvErr,
               "Meta Test Accuracy": testAcc})

def fast_adapt_supcon(batch,
               learner,
               features,
               miner,
               loss_supcon,
               loss_cross,
               adaptation_steps,#5
               shots,#5
               ways,#2
               device=None):

    data, labels = batch
    # data = data[0]#torch.cat([data[0], data[1]], dim=0)
    data, labels = data.to(device), labels.to(device)
    data = features(data)
    #tmpHead = torch.nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim=1), nn.Dropout(0.2))

    # Separate data into adaptation/evaluation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    for step in range(adaptation_steps):
        out = learner(adaptation_data)
        train_error = loss_cross(out, adaptation_labels)
        learner.adapt(train_error)

    # for cross entropy input
    predictions = learner(evaluation_data)

    # for supcon input
    # evaluation_feats = tmpHead(evaluation_data)
    # bsz = int(evaluation_labels.shape[0]/2)
    # f1, f2 = torch.split(evaluation_feats, [bsz, bsz], dim=0)
    # evaluation_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

    # Add supcon loss and cross_entropy loss
    hard_pairs = miner(evaluation_data, evaluation_labels)
    valid_error_sup_conv = loss_supcon(evaluation_data, evaluation_labels, hard_pairs)
    valid_error_cross = loss_cross(predictions, evaluation_labels)

    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error_sup_conv, valid_error_cross, valid_accuracy

def main(args):

    # wandb.login()
    # project_name = args.results_path.split("/")[1]
    # wandb.init(project=project_name, config = args)

    cuda = bool(args.cuda)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda')

    # Create Datasets
    pre_processRGB = transforms.Compose([
                                    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.)),
                                    transforms.RandomHorizontalFlip(),
                                    # transforms.RandomApply([
                                    #     transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
                                    # ], p=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
    train_dataset = customData(img_path=args.data_path,
                                    txt_path=args.txt_path,
                                    data_transforms=pre_processRGB,
                                    phase="meta_train")
    valid_dataset = customData(img_path=args.data_path,
                                    txt_path=args.txt_path,
                                    data_transforms=pre_processRGB,
                                    phase="meta_val")
    
    test_dataset = customData(img_path=args.data_path,
                                    txt_path=args.txt_path,
                                    data_transforms=pre_processRGB,
                                    phase="meta_test")
    
    test_set_dataset = customData(img_path=args.test_data_path,
                                txt_path=args.test_txt_path,
                                    data_transforms=pre_processRGB,
                                    phase="test")

    train_dataset = l2l.data.MetaDataset(train_dataset)
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    test_dataset = l2l.data.MetaDataset(test_dataset)

    train_transforms = [
        FusedNWaysKShots(train_dataset, n=args.ways, k=2*args.shots),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset,
                                       task_transforms=train_transforms,
                                       num_tasks=20000)

    valid_transforms = [
        FusedNWaysKShots(valid_dataset, n=args.ways, k=2*args.shots),
        LoadData(valid_dataset),
        ConsecutiveLabels(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=600)
    
    test_transforms = [
        FusedNWaysKShots(test_dataset, n=args.ways, k=2*args.shots),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
        ConsecutiveLabels(test_dataset),
    ]
    test_tasks = l2l.data.TaskDataset(test_dataset,
                                      task_transforms=test_transforms,
                                      num_tasks=600)

    
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        )
    # Create model

    # features = l2l.vision.models.ConvBase(output_size=64, channels=3, max_pool=True)
    # features = torch.nn.Sequential(features, Lambda(lambda x: x.view(-1, 256)))
    features = FeatExtr()
    
    # features = EfficientNet.from_name("efficientnet-b0")
    features.to(device)

    # head = torch.nn.Linear(256, ways)
    out_feature = features.out_feature
    head = torch.nn.Linear(out_feature, args.ways)
    # head = torch.nn.Linear(128, args.ways)
    # head = torch.nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim=1), nn.Dropout(0.2), head)
    head = l2l.algorithms.MAML(head, lr=args.fast_lr)
    head.to(device)

    # Setup optimization
    all_parameters = list(features.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params=all_parameters, lr=args.meta_lr, betas=(0.9, 0.999), weight_decay=0.01)
    # loss = nn.CrossEntropyLoss(reduction='mean')

    '''
    weights = torch.zeros(args.ways).cuda()
    weights[0]=0.33
    weights[1]=0.66
    reducer = ClassWeightedReducer(weights)#reducers.class_weighted_reducer(weights)
    loss = losses.SupConLoss(temperature=0.1, reducer=reducer)
    '''
    
    loss_cross = nn.CrossEntropyLoss(reduction='mean')
    miner = miners.MultiSimilarityMiner()
    loss_supcon = losses.SupConLoss(temperature=0.1)
    

    # log the weight of model
    # wandb.watch(features, log_freq=100)
    # wandb.watch(head, log_freq=100)

    for iteration in range(args.iters):#20000
        gc.collect()
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        meta_train_error = 0.0
        meta_train_supconv_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_supconv_error = 0.0
        meta_valid_accuracy = 0.0
        meta_test_error = 0.0
        meta_test_supconv_error = 0.0
        meta_test_accuracy = 0.0
        for task in range(args.meta_bsz):#32
            # Compute meta-training loss
            learner = head.clone()
            batch = train_tasks.sample()#20
            valid_error_sup_conv, valid_error_cross, evaluation_accuracy = fast_adapt_supcon(batch,
                                                               learner,
                                                               features,
                                                               miner,
                                                               loss_supcon,
                                                               loss_cross,
                                                               args.adapt_steps,
                                                               args.shots,
                                                               args.ways,
                                                               device)
            evaluation_error = valid_error_sup_conv + valid_error_cross
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_supconv_error += valid_error_sup_conv.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = head.clone()
            batch = valid_tasks.sample()
            valid_error_sup_conv, valid_error_cross, evaluation_accuracy = fast_adapt_supcon(batch,
                                                               learner,
                                                               features,
                                                               miner,
                                                               loss_supcon,
                                                               loss_cross,
                                                               args.adapt_steps,
                                                               args.shots,
                                                               args.ways,
                                                               device)
            evaluation_error = valid_error_sup_conv + valid_error_cross
            meta_valid_error += evaluation_error.item()
            meta_valid_supconv_error += valid_error_sup_conv.item()
            meta_valid_accuracy += evaluation_accuracy.item()

            # Compute meta-testing loss
            learner = head.clone()
            batch = test_tasks.sample()
            valid_error_sup_conv, valid_error_cross, evaluation_accuracy = fast_adapt_supcon(batch,
                                                               learner,
                                                               features,
                                                               miner,
                                                               loss_supcon,
                                                               loss_cross,
                                                               args.adapt_steps,
                                                               args.shots,
                                                               args.ways,
                                                               device)
            evaluation_error = valid_error_sup_conv + valid_error_cross
            meta_test_error += evaluation_error.item()
            meta_test_supconv_error += valid_error_sup_conv.item()
            meta_test_accuracy += evaluation_accuracy.item()
            

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / args.meta_bsz)
        print('Meta Train SupConv Error', meta_train_supconv_error / args.meta_bsz)
        print('Meta Train Accuracy', meta_train_accuracy / args.meta_bsz)
        print('Meta Valid Error', meta_valid_error / args.meta_bsz)
        print('Meta Valid SupConv Error', meta_valid_supconv_error / args.meta_bsz)
        print('Meta Valid Accuracy', meta_valid_accuracy / args.meta_bsz)
        print('Meta Test Error', meta_test_error / args.meta_bsz)
        print('Meta Test SupConv Error', meta_test_supconv_error / args.meta_bsz)
        print('Meta Test Accuracy', meta_test_accuracy / args.meta_bsz)

        log(meta_train_error / args.meta_bsz,
            meta_train_supconv_error / args.meta_bsz,
            meta_train_accuracy / args.meta_bsz,
            meta_valid_error / args.meta_bsz,
            meta_valid_supconv_error / args.meta_bsz,
            meta_valid_accuracy / args.meta_bsz,
            meta_test_error / args.meta_bsz,
            meta_test_supconv_error / args.meta_bsz,
            meta_test_accuracy / args.meta_bsz)

        # Average the accumulated gradients and optimize
        for p in all_parameters:
            if p.grad != None:
                p.grad.data.mul_(1.0 / args.meta_bsz)
        optimizer.step()

        if iteration<100 or iteration%500==0:
            TestSupConv_v2(args, features, learner, test_data_loader, iteration)

        if iteration%1000==0:
            torch.save(features.state_dict(), args.results_path+"Feature-"+str(iteration)+".pt")
            torch.save(learner.state_dict(), args.results_path+"Learner-"+str(iteration)+".pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn2Learn MNIST Example')

    parser.add_argument('--ways', type=int, default=2, metavar='N',
                        help='number of ways (default: 5)')
    parser.add_argument('--shots', type=int, default=5, metavar='N',
                        help='number of shots (default: 1)')
    parser.add_argument('--iters', type=int, default=20000, metavar='N',
                        help='number of iterations (default: 1000)')
    parser.add_argument('--adapt_steps', type=int, default=5, metavar='N',
                        help='number of ways (default: 5)')   
    parser.add_argument('--meta_bsz', type=int, default=32, metavar='N',
                        help='number of ways (default: 5)')                                                

    parser.add_argument('--meta_lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--fast_lr', type=float, default=0.1, metavar='LR',
                        help='learning rate for MAML (default: 0.01)')

    parser.add_argument('--cuda', type=int, default=1, metavar='N',
                        help='number of ways (default: 5)')                                               
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')   
    
    # Path
    parser.add_argument('--data_path', default='../../sinica/CelebA_Data/trainSquareCropped', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument("--txt_path", default='../../sinica/CelebA_Data/metas/intra_test/train_label.txt', type=str)
    parser.add_argument('--results_path', type=str, default="output/check_for_supconv_v2_wo_linear_in_sequtial_wo_norm_w_miner/", metavar='S',
                        help='output dir')
    parser.add_argument('--wandb_name', type=str, default="anil_anti_record_all", metavar='S',
                        help='wandb name')
    # Test
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')   
    parser.add_argument('--test_batch_size', type=int, default=500, metavar='N',
                    help='number of ways (default: 5)')        
    parser.add_argument('--test_data_path', default='../../sinica/CelebA_Data/testSquareCropped', type=str)
    parser.add_argument("--test_txt_path", default='../../sinica/CelebA_Data/metas/intra_test/test_label.txt', type=str)         
    parser.add_argument('--num_workers', default=10, type=int)   
    parser.add_argument('--dataset1', type=str, default='CelebA')
    parser.add_argument('--tstdataset', type=str, default='CelebA') 
    parser.add_argument('--tst_txt_name', type=str, default='norm_testScore.txt')               

    args = parser.parse_args()

    wandb.login()
    wandb.init(project=args.wandb_name, config = args)
    
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path, exist_ok=True)  

    main(args)