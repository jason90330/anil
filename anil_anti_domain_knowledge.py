# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
# from _typeshed import NoneType
import os
import random
import argparse
import wandb
import gc

import numpy as np
import torch
from torch import nn
import torchvision as tv

import learn2learn as l2l
from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels
from efficientnet_pytorch import EfficientNet
# from dataset.customDataColorDepthAnilAnti import customData, customTargetData
from dataset.customDataRgbDepthAnilAnti import customData, customTargetData
from torchvision import transforms
from misc.test import Test
from misc.utils import init_model

from statistics import mean
from copy import deepcopy
import models
torch.nn.Module

class Lambda(nn.Module):

    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def log(trainErr, trainAcc, valErr, valAcc, testErr, testAcc):
    wandb.log({"Meta Train Error": trainErr,
               "Meta Train Accuracy": trainAcc,
               "Meta Valid Error": valErr,
               "Meta Valid Accuracy": valAcc,
               "Meta Test Error": testErr,
               "Meta Test Accuracy": testAcc})

def fast_adapt(batch,
               FeatExtor,
               DepthEstor,
               FeatEmbder,
               Learner,
               criterionCls,
               criterionDepth,
               adaptation_steps,#5
               shots,#5
               ways,#2
               device=None):

    img, labels = batch
    rgb_in, depth_gt = img[0], img[1]
    rgb_in, depth_gt, labels = rgb_in.to(device), depth_gt.to(device), labels.to(device)
    feats = FeatExtor(rgb_in)
    depths = DepthEstor(feats)
    embds = FeatEmbder.extract_features(feats)

    # Separate data into adaptation/evaluation sets
    adaptation_indices = np.zeros(embds.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_rgb_embd, adaptation_labels = embds[adaptation_indices], labels[adaptation_indices]
    evaluation_rgb_embd, evaluation_labels = embds[evaluation_indices], labels[evaluation_indices]

    for step in range(adaptation_steps):
        out = Learner(adaptation_rgb_embd)
        train_error = criterionCls(out, adaptation_labels)
        Learner.adapt(train_error)

    class_predictions = Learner(evaluation_rgb_embd)
    depth_error = criterionDepth(depths, depth_gt)
    valid_error = criterionCls(class_predictions, evaluation_labels)
    valid_accuracy = accuracy(class_predictions, evaluation_labels)
    return depth_error, valid_error, valid_accuracy


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
    pre_processColor = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
    pre_processDepth = transforms.Compose([
                                    transforms.ToTensor()])
    train_dataset = customData(img_path=args.data_path,
                                    txt_path=args.txt_path,
                                    data_transforms_color=pre_processColor,
                                    data_transforms_depth=pre_processDepth,
                                    phase="meta_train")
    valid_dataset = customData(img_path=args.data_path,
                                    txt_path=args.txt_path,
                                    data_transforms_color=pre_processColor,
                                    data_transforms_depth=pre_processDepth,
                                    phase="meta_val")
    
    test_dataset = customData(img_path=args.data_path,
                                    txt_path=args.txt_path,
                                    data_transforms_color=pre_processColor,
                                    data_transforms_depth=pre_processDepth,
                                    phase="meta_test")
    
    test_set_dataset = customTargetData(img_path=args.test_data_path,
                                    txt_path=args.test_txt_path,
                                    data_transforms_color=pre_processColor,
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
    FeatExtmodel = models.create(args.arch_FeatExt, in_ch=3)  
    DepthEstmodel = models.create(args.arch_DepthEst)
    FeatEmbdmodel = models.create(args.arch_FeatEmbd,momentum=args.bn_momentum)
    FeatExtor = init_model(net=FeatExtmodel, init_type = args.init_type, restore=None, parallel_reload=True)
    DepthEstor= init_model(net=DepthEstmodel, init_type = args.init_type, restore=None, parallel_reload=True)
    FeatEmbder= init_model(net=FeatEmbdmodel, init_type = args.init_type, restore=None, parallel_reload=False)

    # features = EfficientNet.from_pretrained("efficientnet-b0", advprop=True)
    # features.to(device)
    FeatExtor.to(device)
    DepthEstor.to(device)
    FeatEmbder.to(device)

    out_feature = FeatEmbder._fc.in_features
    Head = torch.nn.Linear(out_feature, args.ways)
    Head = torch.nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim=1), nn.Dropout(0.2), Head)
    Head = l2l.algorithms.MAML(Head, lr=args.fast_lr)
    Head.to(device)

    # Setup optimization
    # all_parameters = list(features.parameters()) + list(head.parameters())
    # optimizer = torch.optim.Adam(all_parameters, lr=args.meta_lr)
    # optimizer = torch.optim.AdamW(params=all_parameters, lr=args.meta_lr, betas=(0.9, 0.999), weight_decay=0.01)
    embd_params = list(FeatEmbder.parameters())
    head_params = list(Head.parameters())
    extr_params = list(FeatExtor.parameters())
    dep_params = list(DepthEstor.parameters())
    optimizer_embd_head = torch.optim.AdamW(embd_params + head_params,
                                   lr=args.meta_lr,
                                   betas=(0.9, 0.999),
                                   weight_decay=args.weight_decay)

    optimizer_extr = torch.optim.AdamW(extr_params,
                                lr=args.meta_lr,
                                betas=(0.9, 0.999),
                                weight_decay=args.weight_decay)

    optimizer_dep = torch.optim.AdamW(dep_params,
                                lr=args.meta_lr,
                                betas=(0.9, 0.999),
                                weight_decay=args.weight_decay)
    criterionCls = nn.CrossEntropyLoss(reduction='mean')
    criterionDepth = torch.nn.MSELoss()

    # log the weight of model
    # wandb.watch(features, log_freq=100)
    # wandb.watch(head, log_freq=100)

    for iteration in range(args.iters):#20000
        FeatExtor.train(True)
        DepthEstor.train(True)
        FeatEmbder.train(True)
        Head.train(True)
        gc.collect()
        torch.cuda.empty_cache()

        optimizer_embd_head.zero_grad()
        optimizer_extr.zero_grad()
        optimizer_dep.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        for task in range(args.meta_bsz):#32
            # Compute meta-training loss
            Learner = Head.clone()
            batch = train_tasks.sample()#20
            depth_error, evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               FeatExtor,
                                                               DepthEstor,
                                                               FeatEmbder,
                                                               Learner,
                                                               criterionCls,
                                                               criterionDepth,
                                                               args.adapt_steps,
                                                               args.shots,
                                                               args.ways,
                                                               device)
            # evaluation_error.backward()
            evaluation_error.backward(retain_graph=True)
            depth_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            Learner = Head.clone()
            batch = valid_tasks.sample()
            depth_error, evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               FeatExtor,
                                                               DepthEstor,
                                                               FeatEmbder,
                                                               Learner,
                                                               criterionCls,
                                                               criterionDepth,
                                                               args.adapt_steps,
                                                               args.shots,
                                                               args.ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

            # Compute meta-testing loss
            
            Learner = Head.clone()
            batch = test_tasks.sample()
            depth_error, evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               FeatExtor,
                                                               DepthEstor,
                                                               FeatEmbder,
                                                               Learner,
                                                               criterionCls,
                                                               criterionDepth,
                                                               args.adapt_steps,
                                                               args.shots,
                                                               args.ways,
                                                               device)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()
            

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / args.meta_bsz)
        print('Meta Train Accuracy', meta_train_accuracy / args.meta_bsz)
        print('Meta Valid Error', meta_valid_error / args.meta_bsz)
        print('Meta Valid Accuracy', meta_valid_accuracy / args.meta_bsz)
        print('Meta Test Error', meta_test_error / args.meta_bsz)
        print('Meta Test Accuracy', meta_test_accuracy / args.meta_bsz)
        log(meta_train_error / args.meta_bsz,
            meta_train_accuracy / args.meta_bsz,
            meta_valid_error / args.meta_bsz,
            meta_valid_accuracy / args.meta_bsz,
            meta_test_error / args.meta_bsz,
            meta_test_accuracy / args.meta_bsz)

        # Average the accumulated gradients and optimize
        for p in embd_params:
            if p.grad != None:
                p.grad.data.mul_(1.0 / args.meta_bsz)
        for p in head_params:
            if p.grad != None:
                p.grad.data.mul_(1.0 / args.meta_bsz)
        for p in extr_params:
            if p.grad != None:
                p.grad.data.mul_(1.0 / args.meta_bsz)
        for p in dep_params:
            if p.grad != None:
                p.grad.data.mul_(1.0 / args.meta_bsz)
        
        optimizer_embd_head.step()
        optimizer_extr.step()
        optimizer_dep.step()

        if iteration<100 or iteration%500==0:
            Test(args, FeatExtor, DepthEstor, FeatEmbder, Learner, test_data_loader, iteration)

        if (iteration+1)%500==0:
            torch.save(FeatExtor.state_dict(), args.results_path+str(iteration+1)+"_feature.pt")
            torch.save(DepthEstor.state_dict(), args.results_path+str(iteration+1)+"_depth.pt")
            torch.save(FeatEmbder.state_dict(), args.results_path+str(iteration+1)+"_embed.pt")
            torch.save(Learner.state_dict(), args.results_path+str(iteration+1)+"_learner.pt")


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
    parser.add_argument('--weight_decay', type=float, default=0.01, metavar='WD',
                        help='weight decay for MAML (default: 0.01)')

    parser.add_argument('--cuda', type=int, default=1, metavar='N',
                        help='number of ways (default: 5)')                                               
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')   
    
    # Model
    parser.add_argument('--arch_FeatExt', type=str, default='Eff_FeatExtractor')
    parser.add_argument('--arch_DepthEst', type=str, default='Eff_DepthEstmator')
    parser.add_argument('--arch_FeatEmbd', type=str, default='Eff_FeatEmbedder')
    parser.add_argument('--bn_momentum', type=float, default=1)
    parser.add_argument('--init_type', type=str, default='xavier')

    # Path
    parser.add_argument('--data_path', default='../../sinica/CelebA_Data/trainSquareCropped', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument("--txt_path", default='../../sinica/CelebA_Data/metas/intra_test/train_label.txt', type=str)
    parser.add_argument('--results_path', type=str, default="output/anil_anti_domain_w_metatest/", metavar='S',
                        help='output dir')
    # parser.add_argument('--results_path', type=str, default="output/temp/", metavar='S',
    #                     help='output dir')
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
    project_name = args.results_path.split("/")[1]
    wandb.init(project=project_name, config = args)
    
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path, exist_ok=True)  

    main(args)