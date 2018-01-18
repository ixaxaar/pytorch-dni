#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
import torch.nn as nn

from .dni_network import DNI_Network


class RNN_DNI(DNI_Network):

  def __init__(
      self,
      input_size,
      hidden_size,
      output_size,
      num_layers=1,
      batch_first=True,
      dropout=0,
      bias=True,
      kind='lstm'
  ):

    super(RNN_DNI, self).__init__(input_size, hidden_size, output_size)
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.num_layers = num_layers
    self.batch_first = batch_first
    self.dropout = dropout
    self.bias = bias
    self.type = kind

    if self.type == 'lstm':
      self.rnn = nn.LSTM(
          input_size=self.input_size,
          hidden_size=self.hidden_size,
          num_layers=self.num_layers,
          bias=self.bias,
          batch_first=self.batch_first
      )
    elif self.type == 'gru':
      self.rnn = nn.GRU(
          input_size=self.input_size,
          hidden_size=self.hidden_size,
          num_layers=self.num_layers,
          bias=self.bias,
          batch_first=self.batch_first
      )
    elif self.type == 'rnn':
      self.rnn = nn.RNN(
          input_size=self.input_size,
          hidden_size=self.hidden_size,
          num_layers=self.num_layers,
          bias=self.bias,
          batch_first=self.batch_first
      )

    self.output_weights = nn.Linear(self.hidden_size, self.output_size)

  def forward(self, input, hidden=None):
    is_2d = len(list(input.size())) == 2
    if is_2d:
      input = input.unsqueeze(1)

    output, hidden = self.rnn(input, hidden)
    output = self.output_weights(output)

    if is_2d:
      output = output.squeeze()

    return output, hidden
