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
      λ=0,
      recursive=recursive,
      gpu_id=gpu_id
    )
    self.targets = []

    # inherit required private methods
    self.__get_dni_hidden = self._DNI__get_dni_hidden
    self.__save_synthetic_gradient = self._DNI__save_synthetic_gradient

    self._DNI__format = self.__format

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

  def __concat_label(self, output, label):
    if type(output) is T.Size:
      output = cuda(T.randn(output), gpu_id=self.gpu_id)
    if type(label) is T.Size:
      label = cuda(T.randn(label), gpu_id=self.gpu_id)
    if len(label.size()) < len(output.size()):
      label = label.unsqueeze(1).expand_as(output)

    return T.cat([output, label], dim=-1)

  def _forward_update_hook(self):
    def hook(module, input, output):
      log.debug('Forward hook called for ' + str(module))
      output = format(output, module)

      # concat the output and a dummy tensor to represent the final tensor we'd see during training
      concated = self.__concat_label(output, self.target_size)
      # create DNI networks and optimizers if they dont exist (for this module)
      if id(module) not in list(self.dni_networks.keys()):
        self.__create_backward_dni_nets(module, concated, output)

      self.dni_networks_data[id(module)]['input'].append(detach_all(input))

    return hook

  def __format(self, outputs, module):
    output = format(outputs, module)
    concated = self.__concat_label(output, self.target)
    return output, concated

  def _backward_update_hook(self):
    def hook(module, grad_input, grad_output):
      if self.backward_lock:
        log.debug("============= Backward locked for " + str(module))
        return

      log.debug('Backward hook called for ' + str(module) + '  ' +
                str(len(self.dni_networks_data[id(module)]['input'])))

      # store the synthetic gradient output during the backward pass
      handle = module.register_backward_hook(self.__save_synthetic_gradient(module, grad_input))

      self.dni_networks_data[id(module)]['optim'].zero_grad()
      self.dni_networks_data[id(module)]['grad_optim'].zero_grad()

      try:
        input = self.dni_networks_data[id(module)]['input'].pop()
      except IndexError:
        log.warning('Trying to search for non existent output ' + str(module))
        return

      # forward pass through the module alone
      outputs = module(*input)
      output, concated = self.__format(outputs, module)

      # pass through the DNI net
      hx = self.__get_dni_hidden(module)
      predicted_grad, hx = \
          self.dni_networks[id(module)](concated.detach(), hx if hx is None else detach_all(hx))

      # BP(λ)
      predicted_grad = as_type(predicted_grad, grad_output[0])
      if self.λ > 0:
        grad = (1 - self.λ) * predicted_grad + self.λ * grad_output[0]
      else:
        grad = predicted_grad

      self.backward_lock = True
      concated.backward(self.__concat_label(grad, self.target.size()).detach(), retain_graph=True)
      handle.remove()

      # loss is MSE of the estimated gradient (by the DNI network) and the actual gradient
      loss = self.grad_loss(predicted_grad, grad_output[0].detach())

      # backprop and update the DNI net
      loss.backward()

      # track gradient losses
      self.ctr += 1
      self.cumulative_grad_losses = self.cumulative_grad_losses + loss.data.cpu().numpy()[0]
      if self.ctr % 1000 == 0:
        log.info('Average gradient loss last 1k steps: ' + str(self.cumulative_grad_losses / self.ctr))
        self.cumulative_grad_losses = 0
        self.ctr = 0

      # update parameters
      self.dni_networks_data[id(module)]['optim'].step()
      self.dni_networks_data[id(module)]['grad_optim'].step()

      # (back)propagate the (mixed) synthetic and original gradients
      if any(x is None for x in grad_input) or \
              any(x is None for x in self.synthetic_grad_input[id(module)]) or \
              self.synthetic_grad_input[id(module)] is None:
        grad_inputs = None
      else:
        zipped = [(as_type(s, a), a)
                  for s, a in zip(self.synthetic_grad_input[id(module)], grad_input)]
        grad_inputs = tuple(((1 - self.λ) * s) + (self.λ * a.detach()) for (s, a) in zipped)
      return grad_inputs

    return hook

  def forward(self, *args, **kwargs):
    if 'target' in kwargs:
      target = kwargs.pop('target')
      if type(target) is list and self.targets == []:
        self.targets = target
        self.target = self.targets.pop()
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
