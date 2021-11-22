'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - image transforms
    - gradcam
    - misclassification code
    - tensorboard related stuff
    - advanced training policies
'''

import os
import sys
import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations import (pytorch,Normalize,Cutout,Crop)
import matplotlib.pyplot as plt

class Augmentation_TrainDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.mean = self.original_dataset.data.mean()
        self.std = self.original_dataset.data.std()
        self.max = self.original_dataset.data.max()
        self.aug = A.Compose({
        #A.VerticalFlip(p=1),
        #A.PadIfNeeded(min_height=40, min_width=40, always_apply = True),
        A.RandomCrop(height = 32, width = 32, always_apply=False, p=1.0),
        A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, always_apply=False, p=1,fill_value = (.5-self.mean)/self.std)
        #A.Normalize((self.mean,), (self.std,),self.max),
        })
         
    def __len__(self):
        return (len(self.original_dataset))
    
    def __getitem__(self, i):
        data_item = self.original_dataset[i]
        img, lbl = data_item
        aug_img = self.aug(image=np.transpose(np.array(img), (1, 2, 0)))['image']
        #aug_img = self.aug(image = img)['image']
        aug_img = np.transpose(np.array(aug_img), (2,0,1))
        data_item = (aug_img,lbl)    
        return data_item

class Augmentation_TestDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.mean = self.original_dataset.data.mean()
        self.std = self.original_dataset.data.std()
        self.max = self.original_dataset.data.max()
        self.aug = A.Compose({
        #A.Normalize((self.mean,), (self.std,),self.max),
        })
         
    def __len__(self):
        return (len(self.original_dataset))
    
    def __getitem__(self, i):
        data_item = self.original_dataset[i]
        img, lbl = data_item
        aug_img = self.aug(image=np.transpose(np.array(img), (1, 2, 0)))['image']
        aug_img = np.transpose(np.array(aug_img), (2,0,1))
        data_item = (aug_img,lbl)    
        return data_item

def display_graphs(train_lst = [], test_lst = []):
  print('Loss and Accuracy graphs')
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_lst[0])
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_lst[1])
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_lst[0])
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_lst[1])
  axs[1, 1].set_title("Test Accuracy")

def gen_gradcam(p_img_list = None, p_model = None):
  img_lst = []
  gradcam_img = torch.tensor([])
  cpu_model = p_model.to('cpu')
  for i in range(len(p_img_list)):
    pred1 = cpu_model.part_layer1(p_img_list[i].permute(2,0,1).reshape(1,3,32,32))
    pred2 = cpu_model.part_layer2(pred1)
    pred3 = cpu_model.linear(pred2.reshape(-1,512))

    pred3_lbl = pred3.reshape(-1,10).argmax(dim=1).item()
    ## Generate gradient at layer where image size is > 7x7
    pred1.retain_grad()
    # Generate gradient for the wrongly classified image
    pred3[0,pred3_lbl].backward()

    count = 0
    #gradcam_img = torch.tensor([])
    for i in range(len(pred1[0])):
      x = pred1.grad[0,i].mean()
      if x>0 :
        if (count == 0):
          gradcam_img = pred1[0,i] * x
          temp_gradcam = pred1[0,i] * x
        else:
          temp_gradcam = pred1[0,i] * x
          gradcam_img += temp_gradcam
        count += 1
    gradcam_img = gradcam_img/count
    temp_img = gradcam_img.data
    img_lst.append(temp_img)
    #plt.imshow(gradcam_img.data)
    #plt.show()
  if len(gradcam_img) > 0:
    print('\n Gradcam images for above images as below \n')
    for i in range(len(img_lst)):
      plt.subplot(2, 5, i+1)
      plt.axis('off')
      plt.imshow(img_lst[i].squeeze())
  else:
    print('Gradcam not generated...')
    #plt.show()

def find_misclassified_img(p_images = 10, p_model = None, 
                           p_device = 'cpu', p_test_loader = None,
                           p_max = 0, p_mean = 0, p_std = 0):
  print('Misclassified images for Batch Normalization')
  cpu_model = p_model.to(p_device)
  img_counter = 0;
  i = 0;
  img_lst = []
  for test_batch in p_test_loader:
    i += 1
    print(f'Checking batch # {i}')
    #test_batch = next(iter(test_loader)) # Get the first batch from the train loader
    batch_img, batch_lbl = test_batch
    for j in range(len(batch_img)): # processing sets of images in each batch
      img, lbl = batch_img.data[j].to(p_device), batch_lbl.data[j].to(p_device)
      #img, lbl = batch_img.data[j], batch_lbl.data[j]
      #img, lbl = img.to(device), lbl.to(device)
      img1 = img*p_std+p_mean
      test_pred = cpu_model(img.reshape(1,3,32,32)) # passing each test image to the model
      predicted_label = test_pred.reshape(-1,10).argmax(dim=1).item() # finding the max tensor position to determine the predicted label 
      if (predicted_label != lbl): # Check to see if the model prediction was correct and print 10 misclassified images
        img_counter += 1
        plt.imshow(img1.permute(1,2,0))
        plt.show()
        print(f'The model predicted the above image as - {predicted_label}')
        print(f'Actual label of the above image is - {lbl}')

        temp_img = img1.permute(1,2,0)
        img_lst.append(temp_img)
        #plt.subplot(5, 2, 10)
        #plt.axis('off')
        #plt.imshow(img1.numpy().squeeze(), cmap='gray_r')

        if (img_counter > p_images-1):
          break
    if (img_counter > p_images-1):
      print(f'10 misclassified images found.. Processing stopped')
      break
  print ('10 misclassified images as below -  ')
  for i in range(len(img_lst)):
    plt.subplot(2, 5, i+1)
    plt.axis('off')
    plt.imshow(img_lst[i].squeeze())
  plt.show()
  return img_lst

