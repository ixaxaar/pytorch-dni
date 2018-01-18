#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
import torch.nn as nn


class DNI_Network(nn.Module):

  def __init__(
      self,
      input_size,
      hidden_size,
      output_size
  ):

    super(DNI_Network, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

  def forward(self, input, hidden):
    raise Exception('Not implemented')

