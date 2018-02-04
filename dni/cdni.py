#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from .util import *
from .altprop import Altprop
from .dni_nets import LinearDNI
from .dni import DNI
import copy


class CDNI(DNI):

  def __init__(
      self,
      network,
      dni_network=None,
      dni_params={},
      optim=None,
      grad_optim='adam',
      grad_lr=0.001,
      hidden_size=10,
      λ=0,
      target_size=T.Size([1]),
      recursive=True,
      gpu_id=-1
  ):

    assert type(target_size) is T.Size, 'target_size must be of type `torch.Size`'
    self.target_size = target_size

    super(CDNI, self).__init__(
      network=network,
      dni_network=dni_network,
      dni_params=dni_params,
      optim=optim,
      grad_optim=grad_optim,
      grad_lr=grad_lr,
      hidden_size=hidden_size,
      λ=λ,
      recursive=recursive,
      gpu_id=gpu_id
    )
    self.targets = []

    self._DNI__format = self.__format
    log.info('Also conditioning the DNI upon the input')

  def __create_backward_dni_nets(self, module, input, output):
    log.debug('Creating DNI net for ' + str(module))
    # the DNI network
    dni_params = { **self.dni_params, **{'module': module} } \
        if hasattr(self.dni_params, 'module') else self.dni_params
    self.dni_networks[id(module)] = self.dni_network(
        input_size=input.size(-1),
        hidden_size=self.hidden_size,
        output_size=output.size(-1),
        **dni_params
    )
    setattr(self, 'dni_net_' + str(id(module)), self.dni_networks[id(module)])
    log.debug('Created DNI net: \n' + str(self.dni_networks[id(module)]))

    self.dni_networks_data[id(module)] = {}
    # the gradient module's (DNI network) optimizer
    self.dni_networks_data[id(module)]['grad_optim'] = \
        self.get_optim(self.dni_networks[id(module)].parameters(), otype=self.grad_optim, lr=self.lr)

    # the network module's optimizer
    self.dni_networks_data[id(module)]['optim'] = \
        self.get_optim(module.parameters(), otype=self.optim, lr=self.lr)

    # store the DNI outputs (synthetic gradients) here for calculating loss during backprop
    self.dni_networks_data[id(module)]['input'] = []
    self.synthetic_grad_input[id(module)] = None

    if self.gpu_id != -1:
      self.dni_networks[id(module)] = self.dni_networks[id(module)].cuda(self.gpu_id)

  def __adjust_label_size(self, label, ref):
    # expand the labels according the the reference Size
    # so that that can be added with the output or grad
    nr_dims_reqd = len(ref.size())
    nr_dims = len(label.size())

    adjusted = label
    for dim in range(nr_dims_reqd - nr_dims):
      adjusted = adjusted.unsqueeze(1)

    # print(adjusted.shape, label.shape, ref.shape)
    adjusted = adjusted.expand_as(ref)

    return adjusted

  def __add_label(self, output, label):
    if type(output) is T.Size:
      output = cuda(T.randn(output), gpu_id=self.gpu_id)
    if type(label) is T.Size:
      label = cuda(T.randn(label), gpu_id=self.gpu_id)

    if len(label.size()) < len(output.size()):
      label = self.__adjust_label_size(label, output)

    return output + as_type(label, output)

  def __format(self, outputs, module):
    output = format(outputs, module)
    conditioned = self.__add_label(output, self.target)
    return conditioned

  def forward(self, *args, **kwargs):
    if 'target' in kwargs:
      target = kwargs.pop('target')
      if type(target) is list and self.targets == []:
        self.targets = target
        self.target = self.targets.pop()
      elif type(target) is not list:
        self.targets = target
        self.target = self.targets
      else:
        self.target = self.targets.pop()
    else:
      raise ValueError('target not passed, try something like `net(inputs, ..., target=target)`')

    log.debug("=============== Forward pass starting =====================")
    # clear out all buffers
    for i, data in self.dni_networks_data.items():
      data['input'] = []

    self.register_forward(self.network, self._forward_update_hook, recursive=self.recursive)
    ret = self.network(*args, **kwargs)
    self.unregister_forward()
    log.debug("=============== Forward pass done =====================")
    return ret
