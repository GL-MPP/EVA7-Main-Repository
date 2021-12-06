'''Train CIFAR10 with PyTorch.'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import os
import argparse
from tqdm import tqdm
import resnet
from resnet import ResNet18
import custom_resnet
from custom_resnet import Create_Model
import utils
from utils import *

# Data
def prep_data(p_batchsize = 64):
  cuda = torch.cuda.is_available()
  from albumentations import (pytorch,Normalize,Cutout,Crop)
  
  print('==> Downloading data..')
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=None)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=None)

  print('==> Preparing Transforms and Augmentation rules')
  train_transform = transforms.Compose([transforms.ToTensor()
    ,transforms.RandomCrop(size = (32,32),padding = 4,padding_mode='edge')
    ,transforms.RandomHorizontalFlip(p=0.5)
    ])
  
  test_transform = transforms.Compose([transforms.ToTensor()])

  trainset.transform = train_transform  
  testset.transform = test_transform
  
  # The torch vision transforms is followed by albumentation transforms
  augmented_trainset = Augmentation_TrainDataset(trainset)
  augmented_testset = Augmentation_TestDataset(testset)
  #augmented_trainset = trainset
  #augmented_testset = testset

  print('==> Preparing Train/Test loaders')
  dataloader_args = dict(shuffle=True, batch_size=p_batchsize,num_workers=2, pin_memory=True) if cuda else dict (shuffle=True, batch_size=p_batchsize)
  train_loader = torch.utils.data.DataLoader(augmented_trainset, **dataloader_args)
  
  dataloader_args = dict(shuffle=False, batch_size=p_batchsize,num_workers=2, pin_memory=True) if cuda else dict (shuffle=False, batch_size=p_batchsize)
  test_loader = torch.utils.data.DataLoader(augmented_testset, **dataloader_args)
  return train_loader, test_loader,testset


# Initialize model
def initialize_model(p_model = None, p_lr = 0.01,p_momentum = 0.9, p_train_mode = 'N', p_train_loss_lst = [], p_lr_lst = []):
  train_loss=[]
  if (p_train_mode!='N' and (len(p_train_loss_lst)==0 or len(p_lr_lst)==0)):
    print('Error in parameters...Cannot set LR scheduler without MAX LR.\nModel not initialized..')
  else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('==> Initializing Model...')
    model = p_model.to(device)
    print('==> Model initilaized on ',device)
    print('==> Initializing Optimizer...')
    optimizer = optim.Adam(model.parameters(), lr=p_lr)#,momentum = .9)
    print('==> Adam optimizer initialized')
    if(p_train_mode=='N'):
      lr_sch = lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=1.069)
      print('==> LR Scheduler initialized to StepLR to find the max LR with min Loss')
    else:
      train_loss = [x for x in p_train_loss_lst if np.isnan(x) == False]
      train_loss.reverse()
      l_max_lr = p_lr_lst[len(train_loss) - train_loss.index(min(train_loss)) -1]
      print('==> Max LR dervied for Min Loss = ', .09)
      lr_sch = lr_scheduler.OneCycleLR(optimizer,max_lr=.09,pct_start=6/24,total_steps=24,three_phase=False)
      print('==> OCP Scheduler initialized with Max LR -',.09)
    summary(model, input_size=(3, 32, 32))
    return model,optimizer, device, lr_sch


# Train Test Model
def train_model(model, device, train_loader, optimizer, epoch,
                p_train_mode ='N',p_lr_scheduler= None):
  from tqdm import tqdm
  
  criterion = nn.CrossEntropyLoss()
  train_losses = []
  train_acc = []
  lst = []
  lr_lst = []
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  #torch.set_grad_enabled(True)
  model.train()
  #lr = p_starting_lr
  for batch_idx, (data, target) in enumerate(pbar):
    ################################
    ##Find the MAX lr
    if (p_train_mode == 'N'):
      p_lr_scheduler.step()
      lr_lst.append(p_lr_scheduler.get_last_lr())
    #################################
    
        
    data, target = data.to(device), target.to(device)
    # Init
    optimizer.zero_grad()
    y_pred = model(data)
    loss = criterion(y_pred, target)
    train_losses.append(loss.item())
    # Backward
    loss.backward()
    optimizer.step()
    if (loss > 5000):
      exit
    # Update pbar-tqdm
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)
    
    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    
    #if (batch_idx >5):
      #break
  train_acc.append(100*correct/processed)
  
  lst = [train_losses,train_acc,lr_lst]
  return(lst)

def test_model(model, device, test_loader):
  criterion = nn.CrossEntropyLoss()
  test_losses = []
  test_acc = []
  lst = []
  
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
      test_loss += criterion(output, target)#, reduction='sum').item()  # sum up batch loss
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      
      correct += pred.eq(target.view_as(pred)).sum().item()
      
      count += 1
      #if (count > 5):
        #break
      test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_acc.append(100. * correct / len(test_loader.dataset))
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
  
  lst = [test_losses,test_acc]
  return(lst)

def prepare_data(p_batch_size = 64):
  print('==> Preparing Dataset..')
  train_loader,test_loader,test_set = prep_data(p_batchsize=p_batch_size) 
  return train_loader,test_loader,test_set

# Run Train/Test Epoch Loops
def run_train_test_epochs(p_epochs = 1, p_batch_size =1, p_model = None
                        ,p_optimizer = None, p_device = 'cpu'
                        ,p_train_mode = 'N',p_lr_scheduler = None
                        ,p_train_loader = None, p_test_loader = None):
  train_epoch_lst = []
  test_epoch_lst = []
  lr_lst = []
  train_loss = []
  train_acc = []
  test_loss = []
  test_acc = []
  lr_ocp = []
  l_max_lr = 0
  l_min_lr = 0
  l_epoch = 0
  
  if (p_train_mode == 'N'):
    l_epoch = 1 # Run only for 1 epoch to calculate MAX LR
  else:
    l_epoch = p_epochs
  
  
  for epoch in range(l_epoch):
      ## If train mode is N it is being used to find max LR
      if (p_train_mode != 'N'):
        p_lr_scheduler.step()
        print('\n==> Setting One cycle policy LR rate for optimizer. LR set to ',p_lr_scheduler.get_last_lr())

      print("\n==> Train loop for Epoch #:", epoch + 1)
      train_epoch_lst = train_model(p_model, p_device, p_train_loader, p_optimizer, p_epochs
                                  ,p_train_mode=p_train_mode, p_lr_scheduler = p_lr_scheduler)
      ## If train mode is N no need for test validations
      if (p_train_mode != 'N'):
        print("==> Test loop for Epoch #:", epoch)
        test_epoch_lst = test_model(p_model, p_device, p_test_loader)
      train_loss += train_epoch_lst[0]
      train_acc += train_epoch_lst[1]
      lr_lst += train_epoch_lst[2]
      if (p_train_mode != 'N'):
        test_loss += test_epoch_lst[0]
        test_acc += test_epoch_lst[1]
  return train_loss, train_acc,test_loss,test_acc,lr_lst


