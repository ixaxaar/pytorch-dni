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
import functools


class _DNI(Altprop):

  def __init__(
      self,
      network,
      dni_network=None,
      optim=None,
      grad_optim='adam',
      grad_lr=0.001,
      hidden_size=None,
      λ=0.5,
      gpu_id=-1,
      decouple_forwards=False
  ):
    super(_DNI, self).__init__()

    # the DNI network generator
    self.dni_network = LinearDNI if dni_network is None else dni_network

    # the network and optimizer for the entire network
    self.network = network
    self.network_optimizer = \
        optim if optim is not None else self.get_optim(self.network.parameters(), grad_optim, grad_lr)

    # optim params for the DNI networks' optimizers
    self.grad_optim = grad_optim if optim is None else optim.__class__.__name__.lower()
    self.lr = grad_lr if optim is None else optim.defaults['lr']
    # criterion for the grad estimator loss
    self.grad_loss = nn.MSELoss()
    # optim params for the network's per-layer optimizers
    self.optim = str(self.network_optimizer.__class__.__name__).lower()

    # DNI synthetic gradient networks (for each module in network)
    self.grad_dni_networks = {}
    self.grad_dni_network = None
    self.grad_dni_data = {}
    self.input_dni_networks = {}
    self.input_networks_optim = {}

    # lock that prevents the backward and forward hooks to act respectively
    self.backward_lock = False

    # hidden size of the DNI network
    self.hidden_size = 10 if hidden_size is None else hidden_size

    self.λ = λ

    self.gpu_id = gpu_id
    self.decouple_forwards = decouple_forwards

    # register backward hooks to all leaf modules in the network
    monkeypatch_forwards(self.network, self._forward_update_hook)
    self.register_backward(self.network, self._backward_update_hook)
    log.debug(self.network)
    log.debug("=============== Hooks registered =====================")

    log.debug('Using DNI to optimize the network \n' + str(self.network) + '\n with optimizer ' + \
      self.optim + ', lr ' + str(self.lr) + ' with ' + str(self.dni_network.__name__))

    # Set model's methods as our own
    method_list = [m for m in dir(self.network)
                   if callable(getattr(self.network, m)) and not m.startswith("__")
                   and not hasattr(self, m)]
    for m in method_list:
      setattr(self, m, getattr(self.network, m))

  def __get_dni_hidden(self, module):
    # get the DNI network's hidden state
    return self.grad_dni_data[id(module)]['hidden'] \
        if 'hidden' in self.grad_dni_data[id(module)] else None

  def __create_forward_dni_nets(self, module, inputs):
    for ctr, input in enumerate(inputs):
      if id(module) not in self.input_dni_networks:
        self.input_dni_networks[id(module)] = ()
        self.input_networks_optim[id(module)] = ()

      # the input dni network
      net = self.dni_network(
        input_size=input.size(-1),
        hidden_size=self.hidden_size,
        output_size=input.size(-1)
      )
      if self.gpu_id != -1:
        net = net.cuda(self.gpu_id)
      self.input_dni_networks[id(module)] + (net,)

      # optimizer of the input DNI net
      opt = self.get_optim(self.grad_dni_networks[id(module)][ctr].parameters(), otype=self.grad_optim, lr=self.lr)
      self.input_networks_optim + (opt,)

  def __create_backward_dni_nets(self, module, output):
    # the DNI network
    net = self.dni_network(
        input_size=output.size(-1),
        hidden_size=self.hidden_size,
        output_size=output.size(-1)
    )

    self.grad_dni_data[id(module)] = {}
    # the gradient module's (DNI network) optimizer
    # self.grad_dni_data[id(module)]['grad_optim'] = \
    #     self.get_optim(net.parameters(), otype=self.grad_optim, lr=self.lr)

    # the network module's optimizer
    self.grad_dni_data[id(module)]['optim'] = \
        self.get_optim(module.parameters(), otype=self.optim, lr=self.lr)

    # store the DNI outputs (synthetic gradients) here for calculating loss during backprop
    self.grad_dni_data[id(module)]['predicted_grad'] = []
    self.grad_dni_data[id(module)]['grad_input'] = []

    if self.gpu_id != -1:
      net = net.cuda(self.gpu_id)
    self.grad_dni_networks[id(module)] = net

  def __save_synthetic_gradient(self, module):
    def hook(i, o):
      self.grad_dni_data[id(module)]['grad_input'].append(detach_all(i))
    return hook

  def _forward_update_hook(self, forward):
    def hook(*input, **kwargs):
      module = forward.__self__

      if self.training:
        log.debug('Forward called for ' + str(module))

        if id(module) not in list(self.grad_dni_networks.keys()) and self.decouple_forwards:
          self.__create_forward_dni_nets(module, input)

        if id(module) in self.grad_dni_data:
          self.grad_dni_data[id(module)]['optim'].zero_grad()
          # self.grad_dni_data[id(module)]['grad_optim'].zero_grad()

        if self.decouple_forwards:
          # pass through the input dni net
          synthetic = ()
          for ctr, i in enumerate(input):
            si = self.input_dni_networks[id(module)][ctr](i)
            synthetic + (si,)
          gradless_input = synthetic

        # forward pass the input
        gradless_input = detach_all(input)
        outputs = forward(*gradless_input, **kwargs)
        output = format(outputs, module)

        # create grad generating DNI networks and optimizers
        # if they dont exist (for this module)
        if id(module) not in list(self.grad_dni_networks.keys()):
          self.__create_backward_dni_nets(module, output)

        # store the synthetic gradient output during the backward pass
        handle = self.__register_backward_hook(output, self.__save_synthetic_gradient(module))

        # pass through the DNI net
        hx = self.__get_dni_hidden(module)
        predicted_grad, hx = \
          self.grad_dni_networks[id(module)](output.detach(), hx if hx is None else detach_all(hx))

        # backprop through the module and update params
        self.backward_lock = True
        output.backward(predicted_grad.detach())
        self.backward_lock = False
        self.grad_dni_data[id(module)]['optim'].step()
        handle.remove()

        # save predicted_grad for backward
        self.grad_dni_data[id(module)]['predicted_grad'].append(predicted_grad)

      # for a four layer network (three hidden, one final classification) there will be three DNIs.
      # make CNN DNI
      # The spatial resolution of activations from layers in a CNN results in high dimensional activations, so we use synthetic gradient models which themselves are CNNs without pooling and with resolution-preserving zero-padding.
      # change MNIST according to paper

      # is required if we need to backprop (and preserve the graph)
      grad_preserving_outputs = forward(*input, **kwargs)
      return grad_preserving_outputs
    return hook

  def _backward_update_hook(self):
    def hook(module, grad_input, grad_output):
      if self.backward_lock:
        log.debug("============= Backward locked for " + str(module))
        return

      log.debug('Backward hook called for ' + str(module) + '  ' +
            str(len(self.grad_dni_data[id(module)]['predicted_grad'])))

      try:
        predicted_grad = self.grad_dni_data[id(module)]['predicted_grad'].pop()
        synthetic_grad_input = self.grad_dni_data[id(module)]['grad_input'].pop()
      except IndexError:
        log.warning('Trying to search for non existent predicted_grad ' + str(module))
        return

      # loss is MSE of the estimated gradient (by the DNI network) and the actual gradient
      # grad_output is mixed with synthetic gradients of all previous layers (1 + λ + λ^2 + ...)
      loss = self.grad_loss(predicted_grad, grad_output[0].detach())
      self.synthetic_loss = self.synthetic_loss + loss

      # backprop and update the DNI net
      # loss.backward()
      # self.grad_dni_data[id(module)]['grad_optim'].step()

      # BP(λ) the synthetic gradients with the real ones and backprop
      # if synthetic_grad_input != []:
      #   grad_inputs = tuple(((1 - self.λ) * (s if s is not None else a)) + \
      #     (self.λ * a) for s, a in zip(synthetic_grad_input, grad_input))
      # else:
      #   grad_inputs = None
      # return grad_inputs

    return hook

  def cuda(self, device_id):
    self.network.cuda(device_id)
    self.gpu_id = device_id
    return self

  def forward(self, *args, **kwargs):
    log.debug("=============== Forward pass starting =====================")
    # clear out all buffers
    self.synthetic_loss = 0

    ret = self.network(*args, **kwargs)
    if self.grad_dni_network is None:
      self.grad_dni_network = nn.Sequential(*list(self.grad_dni_networks.values()))
      self.grad_dni_network_optim = self.get_optim(self.grad_dni_network.parameters(), otype=self.grad_optim, lr=self.lr)
    log.debug("=============== Forward pass done =====================")
    return ret

  def optimize(self, *args, **kwargs):
    log.debug("=============== Optimizing pass starting =====================")
    self.grad_dni_network_optim.zero_grad()
    self.synthetic_loss.backward()
    self.grad_dni_network_optim.step()
    self.synthetic_loss = 0
    log.debug("=============== Optimizing pass done =====================")
