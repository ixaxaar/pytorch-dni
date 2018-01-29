#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
import torch.nn as nn


class DNINetwork(nn.Module):

  def __init__(
      self,
      input_size,
      hidden_size,
      output_size
  ):

    super(DNINetwork, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

  def forward(self, input, hidden):
    raise Exception('Not implemented')


# default_class_dnis = {
#   'Linear': LinearDNI
#   'Conv2d':
#   'LSTM':
#   'LSTMCell':
#   'GRU':
#   'GRUCell':
#   'RNN':
#   'RNNCell':
# }

# class Network_Dependent_DNI(DNINetwork):

#   def __init__(
#     self,
#     input_size,
#     hidden_size,
#     output_size,
#     **kwargs
#   ):
#     super(Network_Dependent_DNI, self).__init__(input_size, hidden_size, output_size)
#     assert 'module' in kwargs, 'module: parameter not provided'
#     assert 'class_dnis' in kwargs, 'class_dnis: parameter not provided'

#     self.module = kwargs.pop('module')
#     self.class_dnis = kwargs.pop('class_dnis')

#     self.network_params = kwargs

#     class_name = module.__class__.__name__
#     if class_name in self.class_dnis:
#       self.net = self.class_dnis[class_name](
#         input_size=self.input_size,
#         hidden_size=self.hidden_size,
#         output_size=self.output_size,
#         **kwargs
#       )
#     else:
#       self.net = LinearDNI(
#         input_size=self.input_size,
#         hidden_size=self.hidden_size,
#         output_size=self.output_size
#       )

#   def forward(self, input, hidden):
#     return self.net(input, hidden)

