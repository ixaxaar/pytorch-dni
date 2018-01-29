#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
import torch.nn as nn

from .network import DNINetwork


class LinearSigmoidDNI(DNINetwork):

  def __init__(
      self,
      input_size,
      hidden_size,
      output_size,
      num_layers=3,
      bias=True
  ):

    super(LinearSigmoidDNI, self).__init__(input_size, hidden_size, output_size)

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.bias = bias
    self.num_layers = num_layers

    self.net = \
        nn.Sequential(
            nn.Linear(input_size, hidden_size),
            *[nn.Linear(hidden_size, hidden_size) for n in range(self.num_layers-2)],
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

  def forward(self, input, hidden):
    return self.net(input), None
