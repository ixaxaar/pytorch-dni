from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from dni import *
from dni import _DNI

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num-layers', type=int, default=3, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num-filters', type=int, default=32, metavar='N',
                    help='number of filters for the conv layers (default: 32)')
parser.add_argument('--dni-layers', type=str, default="0,1,2", metavar='N',
                    help='layers where to apply fcn (comma-separated string like 0,1,2) (default: "0,1,2")')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
# parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
#           help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--lambda', type=int, default=0.0, metavar='S',
                    help='lambda for DNI: fraction of backprop gradient to mix (default: 0.0)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
  torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

image_size = 28
kernel_size = 5


class Net(nn.Module):

  def __init__(self, num_layers=3, filters=32, dni_layers=[]):
    super(Net, self).__init__()
    self.num_layers = num_layers
    self.filters = filters
    self.padding = get_padding(image_size, kernel_size, 1, 1)

    self.net = [self.dni(self.layer(1 if l == 0 else filters, filters)) if l in dni_layers else self.layer(
        filters) for l in range(self.num_layers)]

    # bind layers to this class (so that they're searchable by pytorch)
    for ctr, n in enumerate(self.net):
      setattr(self, 'layer' + str(ctr), n)

    self.fc1 = nn.Linear(self.filters * image_size * image_size, 50)
    self.fc2 = nn.Linear(50, 10)

  def layer(self, in_filters, out_filters):
    return nn.Sequential(
        nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, padding=self.padding),
        nn.BatchNorm2d(out_filters)
    )

  def dni(self, layer):
    d = DNI(
        layer,
        hidden_size=256,
        dni_network=Conv2dDNI,
        dni_params={'convolutions': self.filters,
                    'kernel_size': kernel_size,
                    'num_layers': 2, 'padding': 'SAME'},
        λ=getattr(args, 'lambda'),
        grad_optim='adam',
        grad_lr=args.lr,
        gpu_id=0 if args.cuda else -1,
        recursive=False
    )
    return d

  def forward(self, input):
    output = input
    for n, layer in enumerate(self.net):
      output = F.relu(F.max_pool2d(layer(output), 2) if n == 0 else F.avg_pool2d(layer(output), 2))

    output = output.view(-1, self.filters * image_size * image_size)
    output = self.fc2(self.fc1(output))
    return F.log_softmax(output, dim=-1)


dni_layers = [int(x) for x in args.dni_layers.split(",")]
model = Net(filters=args.num_filters, num_layers=args.num_layers, dni_layers=dni_layers)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

if args.cuda:
  model.cuda()


def train(epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    if args.cuda:
      data, target = data.cuda(), target.cuda()
    data, target = Variable(data), Variable(target)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    # optimizer.step()
    if batch_idx % args.log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.data[0]))


def test():
  model.eval()
  test_loss = 0
  correct = 0
  for data, target in test_loader:
    if args.cuda:
      data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

  test_loss /= len(test_loader.dataset)
  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
  train(epoch)
  test()
