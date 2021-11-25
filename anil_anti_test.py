import torch
import argparse
from torch import nn
import os
import os.path as osp
from efficientnet_pytorch import EfficientNet
from dataset.customDataAnilAnti import customData
from torchvision import transforms
from misc.utils import init_model, init_random_seed, mkdirs
from misc.test import Test

def main(args):
    cuda = bool(args.cuda)   
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda')

    '''
    Initialize model
    '''
    model = EfficientNet.from_pretrained("efficientnet-b0", advprop=True)
    out_feature = model._fc.in_features
    # head = torch.nn.Linear(256, ways)
    head = torch.nn.Linear(out_feature, args.ways)
    head = torch.nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim=1), nn.Dropout(0.2), head)
    model.to(device)
    head.to(device)

    '''
    Initialize data
    '''
    is_train = False
    # pre_process = transforms.Compose([transforms.ToTensor()])  
    pre_process = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])  
    dataset_test = customData(img_path=args.test_data_path,
                                txt_path=args.test_txt_path,
                                    data_transforms=pre_process,
                                    phase="test")
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        )
    
    iter = 999
    while iter < 20000:
        model_restore = osp.join(args.results_path, str(iter)+'feature.pt')
        head_restore = osp.join(args.results_path, str(iter)+'learner.pt')
        FeatExtor = init_model(net=model, init_type = args.init_type, restore=model_restore, parallel_reload=True)
        head= init_model(net=head, init_type = args.init_type, restore=head_restore, parallel_reload=False)
        '''
        out_feature = FeatExtor._fc.in_features
        # head = torch.nn.Linear(256, ways)
        head = torch.nn.Linear(out_feature, args.ways)
        head = torch.nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim=1), nn.Dropout(0.2), head)        
        checkpoint_head = torch.load(head_restore, map_location='cuda:0')
        head.load_state_dict(checkpoint_head)
        '''
        Test(args, FeatExtor, head, data_loader, iter)
        iter += 1000

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn2Learn MNIST Example')

    parser.add_argument('--cuda', type=int, default=1, metavar='N',
                        help='number of ways (default: 5)')                                               
    
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')   
    
    parser.add_argument('--ways', type=int, default=2, metavar='N',
                        help='number of ways (default: 5)')

    parser.add_argument('--batch_size', type=int, default=800, metavar='N',
                        help='number of ways (default: 5)')        

    parser.add_argument('--test_data_path', default='../../sinica/CelebA_Data/testSquareCropped', type=str)
    
    parser.add_argument("--test_txt_path", default='../../sinica/CelebA_Data/metas/intra_test/test_label.txt', type=str)                        

    parser.add_argument('--results_path', type=str, default="output/anil_anti/", metavar='S',
                        help='output dir')
                        
    parser.add_argument('--num_workers', default=10, type=int)

    parser.add_argument('--init_type', type=str, default='xavier')

    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')                        
    
    parser.add_argument('--dataset1', type=str, default='CelebA')
    
    parser.add_argument('--tstdataset', type=str, default='CelebA')

    parser.add_argument('--tst_txt_name', type=str, default='norm_testScore.txt')
    args = parser.parse_args()                        
    main(args)