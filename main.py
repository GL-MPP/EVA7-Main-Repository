'''Train CIFAR10 with PyTorch.'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import os
import argparse
from tqdm import tqdm
import utils
from utils import *
import resnet
from resnet import ResNet18


# Data
def prep_data(p_batchsize = 64, p_download_required = 'Y'
                           ,p_orig_train_dataset = None
                           ,p_orig_test_dataset = None):
  cuda = torch.cuda.is_available()
  if (p_download_required == 'Y'):
    print('==> Downloading data..')
    train_transform = transforms.Compose([transforms.ToTensor()])

    test_transform = transforms.Compose(
        [transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)

    
    train_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((trainset.data.mean(),),(trainset.data.std(),))])

    test_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((testset.data.mean(),),(testset.data.std(),))])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
             'dog', 'frog', 'horse', 'ship', 'truck')
    augmented_trainset = trainset
    augmented_testset = testset
    dataloader_args = dict(shuffle=True, batch_size=p_batchsize)
    train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)
    dataloader_args = dict(shuffle=False, batch_size=p_batchsize)
    test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)
  else:
    print('==> Preparing augmented Dataset...')
    augmented_trainset = Augmentation_TrainDataset(p_orig_train_dataset)
    augmented_testset = Augmentation_TestDataset(p_orig_test_dataset)

    print('==> Preparing Train/Test loaders')
    dataloader_args = dict(shuffle=True, batch_size=p_batchsize,num_workers=4, pin_memory=True) if cuda else dict (shuffle=True, batch_size=p_batchsize)
    train_loader = torch.utils.data.DataLoader(augmented_trainset, **dataloader_args)
    dataloader_args = dict(shuffle=False, batch_size=p_batchsize,num_workers=4, pin_memory=True) if cuda else dict (shuffle=False, batch_size=p_batchsize)
    test_loader = torch.utils.data.DataLoader(augmented_testset, **dataloader_args)
  return augmented_trainset, augmented_testset,train_loader, test_loader


# Initialize model
def initialize_model(p_lr = 0.01,p_momentum = 0.9):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print('==> Initializing Model...')
  model = ResNet18().to(device)
  print('==> Model initilaized on ',device)
  print('==> Initializing Optimizer...')
  optimizer = optim.SGD(model.parameters(), lr=p_lr, momentum=p_momentum)
  print('==> SGD optimizer initialized')
  summary(model, input_size=(3, 32, 32))
  return model, optimizer, device


# Train Test Model
def train_model(model, device, train_loader, optimizer, epoch):
  from tqdm import tqdm
  criterion = nn.CrossEntropyLoss()
  train_losses = []
  train_acc = []
  lst = []
  lambda_l1 = 0.000001
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  lst = []
  torch.set_grad_enabled(True)
  model.train()
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    
    data, target = data.to(device), target.to(device)
    
    # Init
    optimizer.zero_grad()
    y_pred = model(data)
    loss = criterion(y_pred, target)
    train_losses.append(loss.item())
    # Backpropagation
    loss.backward()
    
    optimizer.step()
    
    # Update pbar-tqdm
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)
    
    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)
    #if (batch_idx >5):
      #break
  lst = [train_losses,train_acc]
  return(lst)

def test_model(model, device, test_loader):
  criterion = nn.CrossEntropyLoss()
  test_losses = []
  test_acc = []
  lst = []
  lambda_l1 = 0.000001
  model.eval()
  test_loss = 0
  correct = 0
  count = 0
  lst = []
  with torch.no_grad():
  #torch.set_grad_enabled(False)
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      
      output = model(data)
      #F.nll_loss
      test_loss += criterion(output, target)#, reduction='sum').item()  # sum up batch loss
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()
      count += 1
      #if (count > 5):
        #break
      test_loss /= len(test_loader.dataset)
      test_losses.append(test_loss)
      test_acc.append(100. * correct / len(test_loader.dataset))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    lst = [test_losses,test_acc]
  return(lst)



# Run Train/Test Epoch Loops
def run_train_test_epochs(p_epochs = 1, p_batch_size =1, p_model = None, p_optimizer = None, p_device = 'cpu'):
  train_epoch_lst = []
  test_epoch_lst = []
  #model = p_model.to(p_device)
  for epoch in range(p_epochs):
    if (epoch == 0):
      orig_train, orig_test, train_loader , test_loader = prep_data(p_batchsize=p_batch_size, p_download_required='Y')
    else:
      print('\n==> Refreshing Dataset for next epoch')
      temp_a,temp_b, train_loader , test_loader = prep_data(p_batchsize=p_batch_size, p_download_required='N',p_orig_train_dataset = orig_train,p_orig_test_dataset = orig_test)
      print("==> Train loop for Epoch #:", epoch)
      train_epoch_lst = train_model(p_model, p_device, train_loader, p_optimizer, p_epochs)
      print("==> Test loop for Epoch #:", epoch)
      test_epoch_lst = test_model(p_model, p_device, test_loader)
    
  return train_epoch_lst, test_epoch_lst


