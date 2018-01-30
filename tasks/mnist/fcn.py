#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
          help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
          help='input batch size for testing (default: 1000)')
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
parser.add_argument('--lambda', type=int, default=0.2, metavar='S',
          help='lambda for DNI: fraction of backprop gradient to mix (default: 0.2)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
          help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
  torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
  datasets.MNIST('../data', train=True, download=True,
           transform=transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))
           ])),
  batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
  datasets.MNIST('../data', train=False, transform=transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))
           ])),
  batch_size=args.test_batch_size, shuffle=True, **kwargs)

image_size = 28

class Net(nn.Module):
  def __init__(self, num_layers=3, hidden_size=256):
    super(Net, self).__init__()
    self.num_layers = num_layers
    self.hidden_size = hidden_size

    self.net = [self.layer(
        image_size*image_size if l == 0 else hidden_size,
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

  def forward(self, x):
    output = x.view(-1, image_size*image_size)
    for layer in self.net:
      output = F.relu(layer(output))
    output = self.final(output)
    return F.log_softmax(output, dim=-1)

model = Net()

optimizer = optim.Adam(model.parameters(), lr=args.lr)

from dni import *
from dni import _DNI

model = DNI(
  model,
  hidden_size=256,
  dni_network=LinearBatchNormDNI,
  λ=getattr(args, 'lambda'),
  grad_optim='adam',
  grad_lr=args.lr
)
if args.cuda:
  model.cuda(0)

def train(epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    if args.cuda:
      data, target = data.cuda(), target.cuda()
    data, target = Variable(data), Variable(target)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    # optimizer.step()
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
