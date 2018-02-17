#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
import torch.nn as nn

class LSTMModel(nn.Module):

  def __init__(self, input_size, hidden_size, num_layers, dropout, batch_first=True):
    super(LSTMModel, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = dropout
    self.batch_first = batch_first

    self.rnn = nn.LSTM(
      self.input_size,
      self.hidden_size,
      num_layers=self.num_layers,
      dropout=self.dropout,
      batch_first=self.batch_first
    )
    self.linear = nn.Linear(self.hidden_size, self.input_size)

  def forward(self, input, hidden=(None, None, None), **kwargs):
    output, chx = self.rnn(input, hidden[0])
    output = self.linear(output)
    return (output, (chx, None, None))


class InhibitoryModel(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, dropout, batch_first=True):
    super(InhibitoryModel, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = dropout
    self.batch_first = batch_first

    self.rnn = nn.LSTM(
      self.input_size,
      self.hidden_size,
      num_layers=self.num_layers,
      dropout=self.dropout,
      batch_first=self.batch_first
    )
    self.linear = nn.Linear(self.hidden_size, 1)

  def forward(self, input, hidden=None):
    output, hx = self.rnn(input, hidden)
    output = self.linear(output)
    return output, hx

  def output(self, o):
    return o[0]

