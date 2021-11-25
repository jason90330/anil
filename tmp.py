import argparse
import random

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from dataset.customData import customData
from dataset.datasets import build_transform
from efficientnet_pytorch import EfficientNet

import learn2learn as l2l


class Net(nn.Module):
    def __init__(self, ways=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc1 = nn.Linear(53*53*50, 10000)
        self.fc2 = nn.Linear(10000, 1000)
        self.fc3 = nn.Linear(1000, ways)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 53*53*50)
        # x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1)
    acc = (predictions == targets).sum().float()
    acc /= len(targets)
    return acc.item()


def main(args, lr=0.005, maml_lr=0.01, iterations=1000, ways=6, shots=1, tps=32, fas=5, device=torch.device("cpu"),
         download_location='~/data'):
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        lambda x: x.view(1, 28, 28),
    ])

    is_train = True
    # dataset_train = customData(img_path=args.data_path,
    #                                 txt_path=args.txt_path,
    #                                 data_transforms=build_transform(is_train, args),
    #                                 phase="train")

    # mnist_train = l2l.data.MetaDataset(dataset_train)

    mnist_train = l2l.data.MetaDataset(MNIST(download_location,
                                             train=True,
                                             download=True,
                                             transform=transformations))

    train_tasks = l2l.data.TaskDataset(mnist_train,
                                       task_transforms=[
                                            l2l.data.transforms.NWays(mnist_train, ways),
                                            l2l.data.transforms.KShots(mnist_train, 2*shots),
                                            l2l.data.transforms.LoadData(mnist_train),
                                            l2l.data.transforms.RemapLabels(mnist_train),
                                            l2l.data.transforms.ConsecutiveLabels(mnist_train),
                                       ],
                                       num_tasks=1000)

    # model = Net(ways)
    model = EfficientNet.from_pretrained("efficientnet-b0", advprop=True, in_channels=1)
    feature = model._fc.in_features
    model._fc = nn.Linear(in_features=feature, out_features=ways, bias=True)
    model.to(device)
    meta_model = l2l.algorithms.MAML(model, lr=maml_lr)
    opt = optim.Adam(meta_model.parameters(), lr=lr)
    loss_func = nn.NLLLoss(reduction='mean')
    # loss_func = nn.BCELoss()

    for iteration in range(iterations):
        iteration_error = 0.0
        iteration_acc = 0.0
        for _ in range(tps):
            learner = meta_model.clone()
            train_task = train_tasks.sample()
            data, labels = train_task
            data = data.to(device)
            labels = labels.to(device)

            # Separate data into adaptation/evalutation sets
            adaptation_indices = np.zeros(data.size(0), dtype=bool)
            adaptation_indices[np.arange(shots*ways) * 2] = True
            evaluation_indices = torch.from_numpy(~adaptation_indices)
            adaptation_indices = torch.from_numpy(adaptation_indices)
            adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
            evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

            # Fast Adaptation
            for step in range(fas):
                out = learner(adaptation_data)
                out = F.log_softmax(out, dim=1)
                train_error = loss_func(out, adaptation_labels)
                train_error.backward(retain_graph=True)
                learner.adapt(train_error)

            # Compute validation loss
            predictions = learner(evaluation_data)
            predictions = F.log_softmax(predictions, dim=1)
            
            valid_error = loss_func(predictions, evaluation_labels)
            valid_error /= len(evaluation_data)
            valid_accuracy = accuracy(predictions, evaluation_labels)
            iteration_error += valid_error
            iteration_acc += valid_accuracy

        iteration_error /= tps
        iteration_acc /= tps
        print('Loss : {:.3f} Acc : {:.3f}'.format(iteration_error.item(), iteration_acc))

        # Take the meta-learning step
        opt.zero_grad()
        iteration_error.backward()
        opt.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn2Learn MNIST Example')

    parser.add_argument('--ways', type=int, default=5, metavar='N',
                        help='number of ways (default: 5)')
    parser.add_argument('--shots', type=int, default=1, metavar='N',
                        help='number of shots (default: 1)')
    parser.add_argument('-tps', '--tasks-per-step', type=int, default=3, metavar='N',
                        help='tasks per step (default: 32)')
    parser.add_argument('-fas', '--fast-adaption-steps', type=int, default=4, metavar='N',
                        help='steps per fast adaption (default: 5)')

    parser.add_argument('--iterations', type=int, default=1000, metavar='N',
                        help='number of iterations (default: 1000)')

    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--maml-lr', type=float, default=0.01, metavar='LR',
                        help='learning rate for MAML (default: 0.01)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--download-location', type=str, default="/tmp/mnist", metavar='S',
                        help='download location for train data (default : /tmp/mnist')

    parser.add_argument('--data_path', type=str, default="../dataset/ok_ng_3rd_edition/", metavar='S',
                        help='download location for train data (default : /tmp/mnist')

    parser.add_argument('--txt_path', type=str, default="", metavar='S',
                        help='download location for train data (default : /tmp/mnist')

    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated_aug', action='store_true')
    parser.add_argument('--no_repeated_aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)


    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    parser.add_argument('--full_crop', action='store_true', help='use crop_ratio=1.0 instead of the\
                                                                  default 0.875 (Used by CaiT).')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")

    main(args,
        lr=args.lr,
        maml_lr=args.maml_lr,
        iterations=args.iterations,
        ways=args.ways,
        shots=args.shots,
        tps=args.tasks_per_step,
        fas=args.fast_adaption_steps,
        device=device,
        download_location=args.download_location)