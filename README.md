# Decoupled Neural Interfaces Using Synthetic Gradients

[![Build Status](https://travis-ci.org/ixaxaar/pytorch-dni.svg?branch=master)](https://travis-ci.org/ixaxaar/pytorch-dni) [![PyPI version](https://badge.fury.io/py/dni.svg)](https://badge.fury.io/py/pytorch-dni)

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

  - [Install](#install)
    - [From source](#from-source)
  - [Architecure](#architecure)
  - [Usage](#usage)

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

```python
from dni import DNI

# Custom network, can be anything extending nn.Module
net = WhateverNetwork(**kwargs)
opt = optim.Adam(net.parameters(), lr=0.001)

# use DNI to optimize this network
net = DNI(net, optim=opt)

# after that we go about our business as usual
for e in range(epoch):
  opt.zero_grad()
  output = net(input, *args)
  loss = criterion(output, target_output)
  loss.backward()
  opt.step()

...
```


