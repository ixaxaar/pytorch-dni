#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
import torch.nn as nn

from .network import DNINetwork
from dni.util import *

class LinearSigmoidDNI(DNINetwork):

  def __init__(
      self,
      input_size,
      hidden_size,
      output_size,
      num_layers=2,
      bias=True
  ):

    super(LinearSigmoidDNI, self).__init__(input_size, hidden_size, output_size)

    self.input_size = input_size
    self.hidden_size = hidden_size * 4
    self.output_size = output_size
    self.num_layers = num_layers
    self.bias = bias

    self.net = [self.layer(
        input_size if l == 0 else self.hidden_size,
        self.hidden_size
    ) for l in range(self.num_layers)]

    # bind layers to this class (so that they're searchable by pytorch)
    for ctr, n in enumerate(self.net):
      setattr(self, 'layer'+str(ctr), n)

    self.final = nn.Linear(self.hidden_size, output_size)

  def layer(self, input_size, hidden_size): return nn.Sequential(
      nn.Linear(input_size, hidden_size),
      nn.Sigmoid()
  )

  def forward(self, input, hidden):
    output = input
    requires_resize = False

    if len(input.size()) > 2:
      requires_resize = True
      b, t = input.size()[:2]
      output = output.contiguous().view(b * t, -1)

    for layer in self.net:
      output = F.relu(layer(output))
    output = self.final(output)

    if len(input.size()) > 2:
      output = output.view(b, t, -1)

    return output, None


class LinearBatchNormDNI(DNINetwork):

  def __init__(
      self,
      input_size,
      hidden_size,
      output_size,
      num_layers=2,
      bias=True
  ):

    super(LinearBatchNormDNI, self).__init__(input_size, hidden_size, output_size)

    self.input_size = input_size
    self.hidden_size = hidden_size * 4
    self.output_size = output_size
    self.num_layers = num_layers
    self.bias = bias

    self.net = [self.layer(
        input_size if l == 0 else self.hidden_size,
        self.hidden_size
    ) for l in range(self.num_layers)]

    # bind layers to this class (so that they're searchable by pytorch)
    for ctr, n in enumerate(self.net):
      setattr(self, 'layer'+str(ctr), n)

    self.final = nn.Linear(self.hidden_size, output_size)

  def layer(self, input_size, hidden_size): return nn.Sequential(
      nn.Linear(input_size, hidden_size),
      nn.BatchNorm1d(hidden_size)
  )

  def forward(self, input, hidden):
    output = input
    requires_resize = False

    if len(input.size()) > 2:
      requires_resize = True
      b, t = input.size()[:2]
      output = output.contiguous().view(b * t, -1)

    for layer in self.net:
      output = F.relu(layer(output))
    output = self.final(output)

    if len(input.size()) > 2:
      output = output.view(b, t, -1)

    return output, None

class LinearDNI(DNINetwork):

  def __init__(
      self,
      input_size,
      hidden_size,
      output_size,
      num_layers=2,
      bias=True
  ):

    super(LinearDNI, self).__init__(input_size, hidden_size, output_size)

    self.input_size = input_size
    self.hidden_size = hidden_size * 4
    self.output_size = output_size
    self.num_layers = num_layers
    self.bias = bias

    self.net = [self.layer(
        input_size if l == 0 else self.hidden_size,
        self.hidden_size
    ) for l in range(self.num_layers)]

    # bind layers to this class (so that they're searchable by pytorch)
    for ctr, n in enumerate(self.net):
      setattr(self, 'layer'+str(ctr), n)

    # final layer (yeah, no kidding)
    self.final = nn.Linear(self.hidden_size, output_size)

  def layer(self, input_size, hidden_size):
      return nn.Linear(input_size, hidden_size)

  def forward(self, input, hidden):
    output = input
    requires_resize = False

    if len(input.size()) > 2:
      requires_resize = True
      b, t = input.size()[:2]
      output = output.contiguous().view(b * t, -1)

    for layer in self.net:
      output = F.relu(layer(output))
    output = self.final(output)

    if len(input.size()) > 2:
      output = output.view(b, t, -1)

    return output, None

