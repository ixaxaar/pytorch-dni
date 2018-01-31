# Decoupled Neural Interfaces Using Synthetic Gradients

[![Build Status](https://travis-ci.org/ixaxaar/pytorch-dni.svg?branch=master)](https://travis-ci.org/ixaxaar/pytorch-dni) [![PyPI version](https://badge.fury.io/py/dni.svg)](https://badge.fury.io/py/pytorch-dni)

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


- [Install](#install)
  - [From source](#from-source)
- [Architecure](#architecure)
- [Usage](#usage)
  - [TLDR: Use DNI to optimize every leaf module](#tldr-use-dni-to-optimize-every-leaf-module-of-net-including-last-layer)
  - [Apply DNI to custom layer](#apply-dni-to-custom-layer)
  - [Apply custom DNI net](#apply-custom-dni-net)
- [DNI Networks](#dni-networks)
- [Custom DNI Networks](#custom-dni-networks)
- [Tasks](#tasks)
- [Notable stuff](#notable-stuff)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

This is an implementation of [Decoupled Neural Interfaces using Synthetic Gradients, Jaderberg et al.](https://arxiv.org/abs/1608.05343).

## Install

```bash
pip install pytorch-dni
```

### From source

```
git clone https://github.com/ixaxaar/pytorch-dni
cd pytorch-dni
pip install -r ./requirements.txt
pip install -e .
```

## Architecure

<img src="./docs/3-6.gif" />

## Usage

### TLDR: Use DNI to optimize every leaf module of `net` (including last layer)

```python
from dni import DNI

# Parent network, can be anything extending nn.Module
net = WhateverNetwork(**kwargs)
opt = optim.Adam(net.parameters(), lr=0.001)

# use DNI to optimize this network
net = DNI(net, grad_optim='adam', grad_lr=0.0001)

# after that we go about our business as usual
for e in range(epoch):
  opt.zero_grad()
  output = net(input, *args)
  loss = criterion(output, target_output)
  loss.backward()

  # Optional: do this to __also__ update net's weight using backprop
  # opt.step()
...
```

### Apply DNI to custom layer

DNI can be applied to any class extending `nn.Module`.
In this example we supply which layers to use DNI for, as the parameter `dni_layers`:

```python

from dni import *

class Net(nn.Module):
  def __init__(self, num_layers=3, hidden_size=256, dni_layers=[]):
    super(Net, self).__init__()
    self.num_layers = num_layers
    self.hidden_size = hidden_size

    self.net = [self.dni(self.layer(
        image_size*image_size if l == 0 else hidden_size,
        hidden_size
    )) if l in dni_layers else self.layer(
        image_size*image_size if l == 0 else hidden_size,
        hidden_size
    ) for l in range(self.num_layers)]
    self.final = self.layer(hidden_size, 10)

    # bind layers to this class (so that they're searchable by pytorch)
    for ctr, n in enumerate(self.net):
      setattr(self, 'layer'+str(ctr), n)

  def layer(self, input_size, hidden_size):
    return nn.Sequential(
      nn.Linear(input_size, hidden_size),
      nn.BatchNorm1d(hidden_size)
    )

  # create a DNI wrapper layer, recursive=False implies treat this layer as a leaf module
  def dni(self, layer):
    d = DNI(layer, hidden_size=256, grad_optim='adam', grad_lr=0.0001, recursive=False)
    return d

  def forward(self, x):
    output = x.view(-1, image_size*image_size)
    for layer in self.net:
      output = F.relu(layer(output))
    output = self.final(output)
    return F.log_softmax(output, dim=-1)

net = Net(num_layers=3, dni_layers=[1,2,3])

# use the gradient descent to optimize layers not optimized by DNI
opt = optim.Adam(net.final.parametes(), lr=0.001)

# after that we go about our business as usual
for e in range(epoch):
  opt.zero_grad()
  output = net(input)
  loss = criterion(output, target_output)
  loss.backward()
```

### Apply custom DNI net

```python
from dni import *

# Custom DNI network
class MyCustomDNI(DNINetwork):

  def __init__(self, input_size, hidden_size, output_size, num_layers=2, bias=True):

    super(LinearDNI, self).__init__(input_size, hidden_size, output_size)

    self.input_size = input_size
    self.hidden_size = hidden_size * 4
    self.output_size = output_size
    self.num_layers = num_layers
    self.bias = bias

    self.net = [self.layer(
        input_size if l == 0 else self.hidden_size,
        self.hidden_size
    ) for l in range(self.num_layers)]

    # bind layers to this class (so that they're searchable by pytorch)
    for ctr, n in enumerate(self.net):
      setattr(self, 'layer'+str(ctr), n)

    # final layer (yeah, no kidding)
    self.final = nn.Linear(self.hidden_size, output_size)

  def layer(self, input_size, hidden_size):
      return nn.Linear(input_size, hidden_size)

  def forward(self, input, hidden):
    output = input
    for layer in self.net:
      output = F.relu(layer(output))
    output = self.final(output)

    return output, None

# Custom network, can be anything extending nn.Module
net = WhateverNetwork(**kwargs)
opt = optim.Adam(net.parameters(), lr=0.001)

# use DNI to optimize this network with MyCustomDNI, pass custom params to the DNI nets
net = DNI(net, grad_optim='adam', grad_lr=0.0001, dni_network=MyCustomDNI,
      dni_params={'num_layers': 3, 'bias': True})

# after that we go about our business as usual
for e in range(epoch):
  opt.zero_grad()
  output = net(input, *args)
  loss = criterion(output, target_output)
  loss.backward()
```

## DNI Networks

This package ships with 3 types of DNI networks:

- [LinearDNI](./dni_nets/linear.py): `Linear -> ReLU` * num_layers -> `Linear`
- [LinearSigmoidDNI](./dni_nets/linear.py): `Linear -> ReLU` * num_layers -> `Linear` -> `Sigmoid`
- [LinearBatchNormDNI](./dni_nets/linear.py): `Linear -> BatchNorm1d -> ReLU` * num_layers -> `Linear`
- [RNNDNI](./dni_nets/rnn.py): stacked `LSTM`s, `GRU`s or `RNN`s
- [Conv2dDNI](./dni_nets/conv.py): `Conv2d -> BatchNorm2d -> MaxPool2d / AvgPool2d -> ReLU` * num_layers -> `Conv2d -> AvgPool2d`

## Custom DNI Networks

Custom DNI nets can be created using the `DNINetwork` interface:

```python
from dni import *

class MyDNI(DNINetwork):
  def __init__(self, input_size, hidden_size, output_size, **kwargs):
    super(MyDNI, self).__init__(input_size, hidden_size, output_size)
    ...

  def forward(self, input, hidden):
    ...
    return output, hidden
```

## Tasks

### MNIST (FCN and CNN)

Refer to [tasks/mnist/README.md](tasks/mnist/README.md)

### Language model

Refer to [tasks/word_language_model/README.md](tasks/word_language_model/README.md)

### Copy task

The tasks included in this project are the same as those in [pytorch-dnc](https://github.com/ixaxaar/pytorch-dnc#tasks), except that they're trained here using DNI.

## Notable stuff

- Using a linear SG module makes the implicit assumption that loss is a quadratic function of the activations
- For best performance one should adapt the SG module architecture to the loss function used. For MSE linear SG is a reasonable choice, however for log loss one should use architectures including a sigmoid applied pointwise to a linear SG
- Learning rates of the order of `1e-5` with momentum of `0.9` works well for rmsprop, adam works well with `0.001`
