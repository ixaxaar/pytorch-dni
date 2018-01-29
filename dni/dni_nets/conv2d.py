# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import torch as T
# import torch.nn as nn

# from .network import DNINetwork


# class Conv_DNI(DNINetwork):

#   def __init__(
#       self,
#       input_size,
#       hidden_size,
#       output_size,
#       convolutions=16,
#       kernel_size=2,
#       stride=1,
#       padding=0,
#       dilation=1,
#       bias=True
#   ):

#     super(Conv_DNI, self).__init__(input_size, hidden_size, output_size)

#     self.input_size = input_size
#     self.hidden_size = hidden_size
#     self.convolutions = convolutions
#     self.output_size = output_size
#     self.kernel_size = kernel_size
#     self.stride = stride
#     self.padding = padding
#     self.dilation = dilation
#     self.bias = bias

#     self.net = \
#         nn.Sequential(
#           nn.Conv2d(1, self.convolutions, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias),
#           # nn.BatchNorm2d(self.convolutions),
#           nn.ReLU(),
#           # nn.MaxPool2d(self.kernel_size)
#         )
#     self.linear = nn.Linear(self.hidden_size, self.output_size)
#     print(self)

#   def forward(self, input, hidden):
#     is_2d = len(list(input.size())) == 2
#     if is_2d:
#       input = input.unsqueeze(1)
#     s = input.size()[:-1]

#     output = self.net(input.unsqueeze(1))
#     b, *_ = output.size()
#     print(input.size(), output.size())
#     output = output.view(b, -1)
#     output = self.linear(output)

#     return output, None
