#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
import torch.nn as nn

from .dni_network import DNI_Network


class Linear_DNI(DNI_Network):

  def __init__(
      self,
      input_size,
      hidden_size,
      output_size,
      bias=True
  ):

    super(Linear_DNI, self).__init__(input_size, hidden_size, output_size)

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.bias = bias

    self.net = \
        nn.Sequential(nn.Linear(input_size, hidden_size), nn.Linear(hidden_size, output_size))

  def forward(self, input, hidden):
    return self.net(input), None
