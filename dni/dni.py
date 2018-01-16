#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from .rnn_dni import RNN_DNI
from .output_format import *

class DNI(nn.Module):

  def __init__(self, network, dni_network=None, optim='adam', hidden_size=None):
    super(DNI, self).__init__()
    self.dni_network = RNN_DNI if dni_network is None else dni_network
    self.network = network
    self.optim = optim
    self.dni_networks = {}
    self.dni_networks_data = {}
    self.forward_hooks = []
    self.backward_hooks = {}
    self.hidden_size = 57 if hidden_size is None else hidden_size

    # register forward hooks to all leaf modules in the network
    self.register_network(self.network, self._forward_update_hook)

  def register_network(self, network, hook):
    for module in network.modules():
      # register hooks only to leaf nodes in the graph with at least 1 learnable Parameter
      l = 0
      for x in module.children(): l += 1
      p = sum([ 1 for x in module.parameters() ])

      if l == 0 and p > 0:
        # register forward and backward hooks
        h = hook()
        print('Registering hooks for ' + str(module))
        module.register_forward_hook(h)
        self.forward_hooks += [{ "name": str(module), "id": id(module), "hook": h }]

  def _forward_update_hook(self):
    def hook(module, input, output):
      # print('Hook called for ' + str(module))
      output = format(output, module)

      # create DNI networks if they dont exist
      if id(module) not in list(self.dni_networks.keys()):
        self.dni_networks[id(module)] = self.dni_network(
          input_size=output.size(-1),
          hidden_size=self.hidden_size,
          output_size=output.size(-1)
        )
        self.dni_networks_data[id(module)] = {}
        self.dni_networks_data[id(module)]['optim'] = \
          self.get_optim(self.dni_networks[id(module)].parameters(), otype=self.optim)

      self.dni_networks_data[id(module)]['optim'].zero_grad()

      # get the DNI network's hidden state
      hx = self.dni_networks_data[id(module)]['hidden'] if 'hidden' in self.dni_networks_data[id(module)] else None

      # pass through the DNI network, get updated gradients for the host network
      grad, hx = self.dni_networks[id(module)](output, None)
      # backprop with generated gradients
      output.backward(grad.detach())
      # parameter = parameter - grad
      self.dni_networks_data[id(module)]['optim'].step()

      # store the hidden state and gradient
      self.dni_networks_data[id(module)]['hidden'] = hx
      self.dni_networks_data[id(module)]['grad'] = grad
    return hook

  def register_network_backward(self, network, hook):
    for module in network.modules():
      # register hooks only to leaf nodes in the graph
      l = 0
      for x in module.children(): l += 1
      if l == 0:
        # register forward and backward hooks
        h = hook(module)
        module.register_backward_hook(h)
        self.backward_hooks += { "name": str(module), "id": id(module), "hook": h }

  def register_network_forward(self, network, hook):
    for module in network.modules():
      # register hooks only to leaf nodes in the graph
      for x in module.children(): l += 1
      if l == 0:
        # register forward and backward hooks
        h = hook(module)
        module.register_forward_hook(h)
        self.backward_hooks += { "name": str(module), "id": id(module), "hook": h }

  def _save_output_hook(self):
    def hook(module, input, output):
      #
      self.output_data.append(output)

  def _backward_update_hook(self, module):
    def hook(grad):
      pass
    return hook

  def forward(self, *kwargs):
    ret = self.network(*kwargs)
    if not self.dni_network:
      self.dni_network = nn.Sequential(*list(self.dni_networks.values()))
      self.dni_optimizer = self.get_optim(self.dni_network.parameters(), otype=self.optim)

  def backward(self, network, optimizer):
    self.dni_optimizer.zero_grad()
    optimizer.zero_grad()

    self.register_network_backward(network, self._backward_update_hook)

  def get_optim(self, parameters, otype="adam", lr=0.001):
    if type(otype) is str:
      if otype == 'adam':
        optimizer = optim.Adam(parameters, lr=lr, eps=1e-9, betas=[0.9, 0.98]) # 0.0001
      elif otype == 'adamax':
        optimizer = optim.Adamax(selfparameters, lr=lr, eps=1e-9, betas=[0.9, 0.98]) # 0.0001
      elif otype == 'rmsprop':
        optimizer = optim.RMSprop(parameters, lr=lr, momentum=0.9, eps=1e-10) # 0.0001
      elif otype == 'sgd':
        optimizer = optim.SGD(parameters, lr=lr) # 0.01
      elif otype == 'adagrad':
        optimizer = optim.Adagrad(parameters, lr=lr)
      elif otype == 'adadelta':
        optimizer = optim.Adadelta(parameters, lr=lr)

    return optimizer


