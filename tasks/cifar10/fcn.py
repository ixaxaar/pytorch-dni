#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

from torch.autograd import Variable

from dni import *
from dni import _DNI

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
          help='input batch size for training (default: 64)')
parser.add_argument('--num-layers', type=int, default=3, metavar='N',
          help='input batch size for training (default: 64)')
parser.add_argument('--dni-layers', type=str, default="0,1,2", metavar='N',
          help='layers where to apply fcn (comma-separated string like 0,1,2) (default: "0,1,2")')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
          help='input batch size for testing (default: 1000)')
parser.add_argument('--hidden-size', type=int, default=1000, metavar='N',
          help='number of epochs to train (default: 100)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
          help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
          help='learning rate (default: 0.0001)')
# parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
#           help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
          help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
          help='random seed (default: 1)')
parser.add_argument('--lambda', type=int, default=0.0, metavar='S',
          help='lambda for DNI: fraction of backprop gradient to mix (default: 0.0)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
          help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
  torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

image_size = 32

class Net(nn.Module):
  def __init__(self, num_layers=3, hidden_size=256, dni_layers=[]):
    super(Net, self).__init__()
    self.num_layers = num_layers
    self.hidden_size = hidden_size

    self.net = [self.dni(self.layer(
        3*image_size*image_size if l == 0 else hidden_size,
        hidden_size
    )) if l in dni_layers else self.layer(
        3*image_size*image_size if l == 0 else hidden_size,
        hidden_size
    ) for l in range(self.num_layers)]
    self.final = self.layer(hidden_size, 10)

    # bind layers to this class (so that they're searchable by pytorch)
    for ctr, n in enumerate(self.net):
      setattr(self, 'layer'+str(ctr), n)

  def layer(self, input_size, hidden_size):
    return nn.Sequential(
      nn.Linear(input_size, hidden_size),
      nn.BatchNorm1d(hidden_size)
    )

  def dni(self, layer):
    d = DNI(
      layer,
      hidden_size=self.hidden_size,
      dni_network=LinearBatchNormDNI,
      λ=getattr(args, 'lambda'),
      grad_optim='adam',
      grad_lr=args.lr,
      gpu_id=0 if args.cuda else -1,
      recursive=False
    )
    return d

  def forward(self, x):
    output = x.view(-1, 3*image_size*image_size)
    for layer in self.net:
      output = F.relu(layer(output))
    output = self.final(output)
    return F.log_softmax(output, dim=-1)


dni_layers = [int(x) for x in args.dni_layers.split(",")] if args.dni_layers != '' else []
model = Net(hidden_size=args.hidden_size ,num_layers=args.num_layers, dni_layers=dni_layers)

final_layer_opt = optim.Adam(model.final.parameters(), lr=args.lr)
non_dni_layers = set(range(model.num_layers)).difference(dni_layers)
non_dni_layers_opt = [optim.Adam(model.net[layer].parameters(), lr=args.lr) for layer in non_dni_layers]

print("DNI layers are", str(dni_layers), "and non dni layers", str(list(non_dni_layers)))

criterion = nn.CrossEntropyLoss()

if args.cuda:
  model.cuda()

def train(epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    if args.cuda:
      data, target = data.cuda(), target.cuda()
    data, target = Variable(data), Variable(target)

    final_layer_opt.zero_grad()
    [ x.zero_grad() for x in non_dni_layers_opt ]

    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    final_layer_opt.step()
    [ x.step() for x in non_dni_layers_opt ]

    if batch_idx % args.log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.data[0]))

def test():
  model.eval()
  test_loss = 0
  correct = 0
  for data, target in test_loader:
    if args.cuda:
      data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

  test_loss /= len(test_loader.dataset)
  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
  train(epoch)
  test()
