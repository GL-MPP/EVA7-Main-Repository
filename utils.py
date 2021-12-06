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
import subprocess
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
from albumentations import (pytorch,Normalize,Cutout,Crop,CoarseDropout)
import matplotlib.pyplot as plt


def install_pkg():
  subprocess.call([sys.executable, "-m", "pip", "install", "multipledispatch"])
  subprocess.call([sys.executable, "-m", "pip", "install", "grad-cam"])
  subprocess.call([sys.executable, "-m", "pip", "install", "cam"])
  subprocess.call([sys.executable, "-m", "pip", "install", "albumentations==0.4.6"])


class Augmentation_TrainDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.mean = self.original_dataset.data.mean()
        self.std = self.original_dataset.data.std()
        self.max = self.original_dataset.data.max()
        self.fill = -1.866605 # Checking for test run
        self.aug = A.Compose({
        A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=self.fill, always_apply=False, p=1),
        A.Normalize(mean = (0.5,), std = (0.5,))
        })
         
    def __len__(self):
        return (len(self.original_dataset))
    
    def __getitem__(self, i):
        data_item = self.original_dataset[i]
        img, lbl = data_item
        self.fill =  img.mean() 
        self.mean = np.array(img.mean()) 
        self.std = np.array(img.std())
        self.max = np.array(img.max()) 
        #self.aug = A.Compose({
        #A.CoarseDropout(max_holes=1, max_height=8, max_width=8, always_apply=False, p=.5,fill_value = self.fill)})
        self.aug = A.Compose({
        A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=self.fill, always_apply=False, p=0.5),
        A.Normalize(mean = (self.mean,), std = (self.std,), max_pixel_value = self.max)
        })
        aug_img = self.aug(image=np.transpose(np.array(img), (1, 2, 0)))['image']
        aug_img = np.transpose(np.array(aug_img), (2,0,1))
        aug_img = torch.tensor(aug_img)
        data_item = (aug_img,lbl)    
        return data_item

class Augmentation_TestDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.mean = self.original_dataset.data.mean()
        self.std = self.original_dataset.data.std()
        self.max = self.original_dataset.data.max()
        self.aug = A.Compose({
        A.Normalize((self.mean,), (self.std,),self.max)
        })
         
    def __len__(self):
        return (len(self.original_dataset))
    
    def __getitem__(self, i):
        data_item = self.original_dataset[i]
        img, lbl = data_item
        self.mean = np.array(img.mean()) 
        self.std = np.array(img.std())
        self.max = np.array(img.max())
        self.aug = A.Compose({
        A.Normalize(mean = (self.mean,), std = (self.std,), max_pixel_value = self.max)
        })
        aug_img = self.aug(image=np.transpose(np.array(img), (1, 2, 0)))['image']
        aug_img = np.transpose(np.array(aug_img), (2,0,1))
        aug_img = torch.tensor(aug_img)
        data_item = (aug_img,lbl)    
        return data_item

def display_graphs(train_loss,train_acc,test_loss,test_acc,lr_range):
    
  if (len(lr_range)>0):
    plt.plot(np.log(lr_range),train_loss)
    plt.xlabel('LR')
    plt.ylabel('Training Loss')
    plt.title('LR vs Training Loss')
    plt.show()
  else:
    print('Loss and Accuracy graphs')
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_loss)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_loss)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    plt.show()
    print('LR Batch Test')
    #fig, axs = plt.subplots(figsize=(15,10))

def gen_heatmap (p_img_list = None, p_model = None, p_target_layers = None,p_category = 1):
  import cam
  from pytorch_grad_cam.base_cam import BaseCAM
  import pytorch_grad_cam
  from pytorch_grad_cam import GradCAM
  hm_img_lst = []
  #cpu_model = p_model.to('cpu')
  cam = GradCAM(model=p_model, target_layers=p_target_layers)
  for i in range(len(p_img_list)):
    input_img = torch.tensor(p_img_list[i])
    input_img = input_img.reshape(1,3,32,32)
    hm_img = cam(input_tensor= input_img,target_category= p_category)
    hm_img = torch.tensor(hm_img)
    hm_img_lst.append(hm_img.reshape(32,32))
    #gen_heatmap(img_lst,model,model.part_layer1)
  if len(hm_img_lst) > 0:
    print('\n Gradcam images for above images as below \n')
    for i in range(len(hm_img_lst)):
      plt.subplot(2, 5, i+1)
      plt.axis('off')
      plt.imshow(hm_img_lst[i])
  else:
    print('Gradcam not generated...')

def gen_gradcam(p_img_list = None, p_model = None):
  img_lst = []
  
  # Custom code
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
                           p_device = 'cpu', p_test_loader = None,p_test_set = None):
  print('Misclassified images for Batch Normalization')
  cpu_model = p_model.to(p_device)
  img_counter = 0;
  i = 0;
  img_lst=[]
  ret_img_lst=[]
  for test_batch in p_test_loader:
    print(f'Checking batch # {i+1}')
    #test_batch = next(iter(test_loader)) # Get the first batch from the train loader
    batch_img, batch_lbl = test_batch
    for j in range(len(batch_img)): # processing sets of images in each batch
      img, lbl = batch_img.data[j].to(p_device), batch_lbl.data[j].to(p_device)
      img1 = p_test_set.data[i+j] 
      test_pred = cpu_model(img.reshape(1,3,32,32)) # passing each test image to the model
      predicted_label = test_pred.reshape(-1,10).argmax(dim=1).item() # finding the max tensor position to determine the predicted label 
      if (predicted_label != lbl): # Check to see if the model prediction was correct and print 10 misclassified images
        img_counter += 1
        plt.figure(figsize = (2,2))
        plt.imshow(img1)
        plt.show()
        print(f'The model predicted the above image as - {p_test_set.classes[predicted_label]}')
        print(f'Actual label of the above image is - {p_test_set.classes[lbl]}')

        temp_img = img1
        img_lst.append(torch.tensor(temp_img))
        ret_img_lst.append(img)

        if (img_counter > p_images-1):
          break
    i+=1
    if (img_counter > p_images-1):
      print(f'10 misclassified images found.. Processing stopped')
      break
  print ('10 misclassified images as below -  ')
  for i in range(len(img_lst)):
    plt.subplot(2, 5, i+1)
    plt.axis('off')
    plt.imshow(img_lst[i].squeeze())
  plt.show()
  return ret_img_lst


