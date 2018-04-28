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


class LDNI(Altprop):

  def __init__(
      self,
      network,
      grad_network=LinearDNI,
      optim=None,
      optim_type='adam',
      lr=0.001,
      grad_params={},
      hidden_size=None,
      recursive=True,
      gpu_id=-1
  ):
    super(LDNI, self).__init__()

    # the parent network
    self.network = network
    self.grad_network = grad_network
    self.grad_params = grad_params
    self.hidden_size = 10 if hidden_size is None else hidden_size

    # lock that prevents the backward and forward hooks to act respectively
    self.backward_lock = False

    # optim params for the network's per-layer optimizers
    self.lr = lr if optim is None else optim.defaults['lr']
    self.optim = optim_type if optim is None else optim.__class__.__name__.lower()
    self.grad_optim = self.optim
    self.grad_lr = self.lr
    self.loss = None
    # criterion for the grad estimator loss
    self.grad_loss = nn.MSELoss()

    # whether we should apply triggers recursively
    self.recursive = recursive

    self.gpu_id = gpu_id
    self.ctr = 0
    self.cumulative_grad_losses = 0

    self.grad_nets = {}
    self.network_data = {}

    # register backward hooks to all leaf modules in the network
    self.register_backward(self.network, self._backward_update_hook, recursive=self.recursive)
    log.debug(self.network)
    log.debug("=============== Hooks registered =====================")

    log.info('Using gradient estimators to optimize the network \n' + str(self.network) + '\n with optimizer ' +
             self.optim + ', lr ' + str(self.lr))

    # Set model's methods as our own
    method_list = [m for m in dir(self.network)
                   if callable(getattr(self.network, m)) and not m.startswith("__")
                   and not hasattr(self, m)]
    for m in method_list:
      setattr(self, m, getattr(self.network, m))

  def __get_grad_hidden(self, module):
    # get the grad network's hidden state
    return self.network_data[id(module)]['hidden'] \
        if 'hidden' in self.network_data[id(module)] else None

  def __create_grad_nets(self, module, output):
    log.debug('Creating grad net for ' + str(module))
    # the grad network
    grad_params = { **self.grad_params, **{'module': module} } \
        if hasattr(self.grad_params, 'module') else self.grad_params

    self.grad_nets[id(module)] = self.grad_network(
        input_size=output.size(-1) + 1,
        hidden_size=self.hidden_size,
        output_size=output.size(-1),
        **grad_params
    )
    setattr(self, 'grad_net_' + str(id(module)), self.grad_nets[id(module)])
    log.debug('Created grad net: \n' + str(self.grad_nets[id(module)]))

    self.network_data[id(module)] = {}
    # the gradient module's (grad network) optimizer
    self.network_data[id(module)]['grad_optim'] = \
        self.get_optim(self.grad_nets[id(module)].parameters(), otype=self.grad_optim, lr=self.lr)

    # the network module's optimizer
    self.network_data[id(module)]['optim'] = \
        self.get_optim(module.parameters(), otype=self.optim, lr=self.lr)

    self.network_data[id(module)]['input'] = []

    if self.gpu_id != -1:
      self.grad_nets[id(module)] = self.grad_nets[id(module)].cuda(self.gpu_id)


  def __format(self, outputs, module):
    return format(outputs, module)

  def _forward_update_hook(self):
    def hook(module, input, output):
      log.debug('Forward hook called for ' + str(module))
      output = format(output, module)

      # create grad networks and optimizers if they dont exist (for this module)
      if id(module) not in list(self.grad_nets.keys()):
        self.__create_grad_nets(module, output)

      if self.training:
        self.network_data[id(module)]['input'].append(detach_all(input))

    return hook

  def _backward_update_hook(self):
    def hook(module, grad_input, grad_output):
      if self.backward_lock:
        log.debug("============= Backward locked for " + str(module))
        return

      log.debug('Backward hook called for ' + str(module) + '  ' +
                str(len(self.network_data[id(module)]['input'])))

      self.network_data[id(module)]['optim'].zero_grad()
      self.network_data[id(module)]['grad_optim'].zero_grad()

      try:
        input = self.network_data[id(module)]['input'].pop()
      except IndexError:
        log.debug('Trying to search for non existent output ' + str(module))
        return

      # forward pass through the module alone
      outputs = module(*input)
      output = self.__format(outputs, module)

      loss = self.loss
      output_shape = output.size()
      for dim in range(len(output_shape)-1):
        loss = loss.unsqueeze(0)
      loss = self.loss.expand_as(T.zeros([*output_shape[:-1], 1]))

      # pass through the grad net
      hx = self.__get_grad_hidden(module)
      concated = T.cat([output, loss], -1)
      grad, hx = self.grad_nets[id(module)](concated, hx)
      self.network_data[id(module)]['hidden'] = hx

      self.backward_lock = True
      output.backward(grad.detach())
      self.backward_lock = False

      grad_loss = self.grad_loss(grad, grad_output[0])
      grad_loss.backward()

      # update parameters
      self.network_data[id(module)]['optim'].step()
      self.network_data[id(module)]['grad_optim'].step()

    return hook

  def forward(self, *args, **kwargs):
    log.debug("=============== Forward pass starting =====================")
    # clear out all buffers
    self.loss = None
    for i, data in self.network_data.items():
      data['input'] = []

    self.register_forward(self.network, self._forward_update_hook, recursive=self.recursive)
    ret = self.network(*args, **kwargs)
    self.unregister_forward()
    log.debug("=============== Forward pass done =====================")
    return ret

  def register_loss(self, loss):
    self.loss = loss.detach()

  def cuda(self, device_id=0):
    self.network = self.network.cuda(device_id)
    self.grad_nets = { k: v.cuda(device_id) for k,v in self.grad_nets.items()}
    self.gpu_id = device_id
    return self
