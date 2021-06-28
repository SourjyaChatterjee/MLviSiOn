import numpy as np
import pandas as pd
import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler




#######################

## Data Augmentation ##

#######################


def train_valid_transform(X):
  train_transforms = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(15),                               
                                transforms.ToTensor(),
                                transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
 
  X_trans = train_transforms(X)
  return X_trans


def test_transform(X):
  test_transforms = transforms.Compose([
                                transforms.ToPILImage(),                               
                                transforms.ToTensor(),
                                transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
  X_trans = test_transforms(X)
  return X_trans





#######################

## Custom DataLoader ##

#######################


class MyDataset():
  
  def __init__(self,image_set, mode = 'train'):

    data = pd.read_csv(image_set)
    self.imgfiles = list(data['images'])
    self.annotations = list(data['annotations'])
    self.mode = mode
 
  def __len__(self):
    return len(self.imgfiles)
 
  def __getitem__(self,idx):

    img = cv2.imread(self.imgfiles[idx])
    X = np.asarray(img,dtype=np.uint8)
    annot = cv2.imread(self.annotations[idx])
    Y = np.sum(np.round(annot/255))

    if self.mode == 'test':
      X = test_transform(X)
    else:
      X = train_valid_transform(X)

    return X.double(),Y





#######################

## VGG Data Loading ##

#######################


def Data_Loader(vgg_train_data, vgg_test_data,batch_size):

  valid_size = 0.2
  indices = list(range(len(vgg_train_data)))
  np.random.shuffle(indices)
  split = int(np.floor((valid_size * len(vgg_train_data))))
  vgg_valid_idx , vgg_train_idx = indices[:split] , indices[split:]

  vgg_train_sampler = SubsetRandomSampler(vgg_train_idx)
  vgg_valid_sampler = SubsetRandomSampler(vgg_valid_idx)

  vgg_train_loader = DataLoader(vgg_train_data,batch_size=batch_size,sampler=vgg_train_sampler)
  vgg_valid_loader = DataLoader(vgg_train_data,batch_size=batch_size,sampler=vgg_valid_sampler)
  vgg_test_loader = DataLoader(vgg_test_data, batch_size = 1 , shuffle = True)    

  return vgg_train_loader, vgg_valid_loader, vgg_test_loader