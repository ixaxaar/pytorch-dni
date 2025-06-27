#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from .util import *
import copy


class Altprop(nn.Module):

  def __init__(self):
    super(Altprop, self).__init__()
    self.forward_hooks = []
    self.backward_hooks = []

  def register_forward(self, network, hook, recursive=True, skip_last_layer=True):
    if skip_last_layer:
      last_layer = self.__find_last_layer(network)

    # apply hook to supplied network only
    if not recursive:
      h = hook()
      handle = network.register_forward_hook(h)
      self.forward_hooks += [{"name": str(network), "id": id(network), "hook": handle}]
    # recursively apply hook to leaf modules of the network
    else:
      for module in network.modules():
        # register hooks only to leaf nodes in the graph with at least 1 learnable Parameter
        children = sum([1 for x in module.children()])
        params = sum([1 for x in module.parameters()])

        if children == 0 and params > 0:
          if skip_last_layer and id(module) == id(last_layer):
            return
          # register forward hooks
          h = hook()
          log.debug('Registering forward hooks for ' + str(module))
          handle = module.register_forward_hook(h)
          self.forward_hooks += [{"name": str(module), "id": id(module), "hook": handle}]

  def unregister_forward(self):
    for h in self.forward_hooks:
      h['hook'].remove()
    self.forward_hooks = []

  def unregister_backward(self):
    for h in self.backward_hooks:
      h['hook'].remove()
    self.backward_hooks = []

  def __find_last_layer(self, network):
    # Find out the last layer / leaf module module
    last_layer = None
    for module in network.modules():
      children = sum([1 for x in module.children()])
      params = sum([1 for x in module.parameters()])
      if children == 0 and params > 0:
        last_layer = module
    return last_layer

  def register_backward(self, network, hook, recursive=True, skip_last_layer=True):
    if skip_last_layer:
      last_layer = self.__find_last_layer(network)

    # apply hook to supplied network only
    if not recursive:
      h = hook()
      handle = network.register_backward_hook(h)
      self.backward_hooks += [{"name": str(network), "id": id(network), "hook": handle}]
    # recursively apply hook to leaf modules of the network
    else:
      for module in network.modules():
        # register hooks only to leaf nodes in the graph with at least 1 learnable Parameter
        children = sum([1 for x in module.children()])
        params = sum([1 for x in module.parameters()])

        if children == 0 and params > 0:
          if skip_last_layer and id(module) == id(last_layer):
            return last_layer
          # register backward hooks
          h = hook()
          log.debug('Registering backward hooks for ' + str(module))
          module.register_backward_hook(h)
          self.backward_hooks += [{"name": str(module), "id": id(module), "hook": h}]

  def __register_backward_hook(self, variable, hook):
    # for other hooks this is done in __call__ before forward
    if hasattr(variable, 'grad_fn'):
      grad_fn = variable.grad_fn
      # print(dir(grad_fn), grad_fn.next_functions)
      if grad_fn is not None:
        return grad_fn.register_hook(hook)

  def get_optim(self, parameters, otype="adam", lr=0.001):
    if type(otype) is str:
      if otype == 'adam':
        optimizer = optim.Adam(parameters, lr=lr, eps=1e-9, betas=[0.9, 0.98])
      elif otype == 'adamax':
        optimizer = optim.Adamax(parameters, lr=lr, eps=1e-9, betas=[0.9, 0.98])
      elif otype == 'rmsprop':
        optimizer = optim.RMSprop(parameters, lr=lr, momentum=0.9, eps=1e-10)
      elif otype == 'sgd':
        optimizer = optim.SGD(parameters, lr=lr)  # 0.01
      elif otype == 'adagrad':
        optimizer = optim.Adagrad(parameters, lr=lr)
      elif otype == 'adadelta':
        optimizer = optim.Adadelta(parameters, lr=lr)

    return optimizer
