# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

import pytest
import numpy as np

import sys
import os
import math
import time
import functools
sys.path.insert(0, '.')

from dnc import SAM
from dni.util import detach_all

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from dni import *

from collections import defaultdict

def test_one_big_function():
  arglist = [
    ('batch_size', 10),
    ('num_layers', 3),
    ('num_filters', 32),
    ('dni_layers', '0,1,2'),
    ('cdni', True),
    ('test_batch_size', 10),
    ('epochs', 1),
    ('lr', 0.001),
    ('cuda', False),
    ('seed', 1111),
    ('lambda', 0.0),
    ('log_interval', 100)
  ]
  args = dict(arglist)


  torch.manual_seed(args['seed'])

  kwargs = {'num_workers': 1, 'pin_memory': True} if args['cuda'] else {}
  train_loader = torch.utils.data.DataLoader(
      datasets.MNIST('../data', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))
                     ])),
      batch_size=args['batch_size'], shuffle=True, **kwargs)
  test_loader = torch.utils.data.DataLoader(
      datasets.MNIST('../data', train=False, transform=transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.1307,), (0.3081,))
      ])),
      batch_size=args['test_batch_size'], shuffle=True, **kwargs)

  image_size = 28
  kernel_size = 5

  class Net(nn.Module):

    def __init__(self, num_layers=3, filters=32, dni_layers=[]):
      super(Net, self).__init__()
      self.num_layers = num_layers
      self.filters = filters
      self.dni_layers = dni_layers
      self.padding = get_padding(image_size, kernel_size, 1, 1)

      self.net = [self.dni(self.layer(1 if l == 0 else filters, filters)) if l in dni_layers else self.layer(
          1 if l == 0 else filters, filters) for l in range(self.num_layers)]

      # bind layers to this class (so that they're searchable by pytorch)
      for ctr, n in enumerate(self.net):
        setattr(self, 'layer' + str(ctr), n)

      self.final = nn.Sequential(
          nn.Linear(self.filters * image_size * image_size, 50),
          nn.Linear(50, 10)
      )

    def layer(self, in_filters, out_filters):
      return nn.Sequential(
          nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, padding=self.padding),
          nn.BatchNorm2d(out_filters)
      )

    def dni(self, layer):
      if args['cdni']:
        d = CDNI(
            layer,
            hidden_size=256,
            dni_network=Conv2dDNI,
            dni_params={'convolutions': self.filters,
                        'kernel_size': kernel_size,
                        'num_layers': 2, 'padding': 'SAME'},
            λ=args['lambda'],
            grad_optim='adam',
            grad_lr=args['lr'],
            gpu_id=0 if args['cuda'] else -1,
            recursive=False,
            target_size=torch.Size([args['batch_size'], image_size, image_size])
        )
      else:
        d = DNI(
              layer,
              hidden_size=256,
              dni_network=Conv2dDNI,
              dni_params={'convolutions': self.filters,
                          'kernel_size': kernel_size,
                          'num_layers': 2, 'padding': 'SAME'},
              λ=getattr(args, 'lambda'),
              grad_optim='adam',
              grad_lr=args['lr'],
              gpu_id=0 if args['cuda'] else -1,
              recursive=False
          )
      return d

    def forward(self, input, target=None):
      output = input
      for n, layer in enumerate(self.net):
        if args['cdni']:
          output = F.relu(F.max_pool2d(layer(output, target=target), 2) if n == 0 and n in self.dni_layers
                        else F.avg_pool2d(layer(output, target=target), 2))
        else:
          output = F.relu(F.max_pool2d(layer(output), 2) if n == 0 and n in self.dni_layers
                          else F.avg_pool2d(layer(output), 2))

      output = output.view(-1, self.filters * image_size * image_size)
      output = self.final(output)
      return F.log_softmax(output, dim=-1)


  dni_layers = [int(x) for x in args['dni_layers'].split(",")] if args['dni_layers'] != '' else []
  model = Net(filters=args['num_filters'], num_layers=args['num_layers'], dni_layers=dni_layers)

  final_layer_opt = optim.Adam(model.final.parameters(), lr=args['lr'])
  non_dni_layers = set(range(model.num_layers)).difference(dni_layers)
  non_dni_layers_opt = [optim.Adam(model.net[layer].parameters(), lr=args['lr']) for layer in non_dni_layers]


  if args['cuda']:
    model.cuda()

  def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
      if args['cuda']:
        data, target = data.cuda(), target.cuda()
      data, target = Variable(data), Variable(target)

      final_layer_opt.zero_grad()
      [x.zero_grad() for x in non_dni_layers_opt]

      output = model(data, target=target)
      loss = F.nll_loss(output, target)
      loss.backward()

      final_layer_opt.step()
      [x.step() for x in non_dni_layers_opt]

      if batch_idx % args['log_interval'] == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data[0]))
        return

  train(1)