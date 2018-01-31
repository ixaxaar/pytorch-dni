#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from .network import DNINetwork
from dni.util import *


class Conv2dDNI(DNINetwork):

  def __init__(
      self,
      input_size,
      hidden_size,
      output_size,
      convolutions=16,
      kernel_size=2,
      stride=1,
      padding=0,
      dilation=1,
      num_layers=2,
      bias=True
  ):

    super(Conv2dDNI, self).__init__(input_size, hidden_size, output_size)

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.convolutions = convolutions
    self.output_size = output_size
    self.kernel_size = kernel_size
    self.stride = stride
    self.dilation = dilation
    self.num_layers = num_layers
    self.bias = bias

    # get dimension preserving padding
    self.padding = get_padding(self.input_size,
                               self.kernel_size,
                               self.dilation,
                               self.stride) if padding == 'SAME' else padding

    self.net = [self.layer(self.convolutions, self.convolutions)
                for l in range(self.num_layers)]

    # bind layers to this class (so that they're searchable by pytorch)
    for ctr, n in enumerate(self.net):
      setattr(self, 'layer' + str(ctr), n)

    self.final = self._conv(self.convolutions, self.convolutions)

  def _conv(self, in_filters, out_filters, padding=-1):
    return nn.Conv2d(
        in_filters,
        out_filters,
        kernel_size=self.kernel_size,
        stride=self.stride,
        dilation=self.dilation,
        padding=self.padding
    )

  def layer(self, in_filters, out_filters):
    return nn.Sequential(
        self._conv(in_filters, out_filters),
        nn.BatchNorm2d(out_filters)
    )

  def forward(self, input, hidden):
    output = input
    for n, layer in enumerate(self.net):
      output = F.relu(F.max_pool2d(layer(output), 2) if n == 0 else F.avg_pool2d(layer(output), 2))

    output = F.avg_pool2d(self.final(output), 2)
    return output, None
