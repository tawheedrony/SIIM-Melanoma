import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchvision import transforms,models
from torch.utils.data import Dataset,DataLoader
import os
from sklearn import model_selection
from CustomDataset import ClassificationDataset
import albumentations

# file directory
root_dir = 'D:/Deep Learning/Projects/Melanoma skin cancer detection/'

# not using
# Folding the validation set using stratified cross validation . Stratified cross validation works well for skewed dataset
if __name__ == "__main__":
    train_csv = pd.read_csv(root_dir +'train.csv')
    train_csv['kfold'] = -1
    # DataFrame.sample randomizes the rows of the data and frac determines the fraction of the data to be used for randomizing
    # DataFrame.reset_index creates new index for the randomized data and drop=True delete the previous index
    train_csv = train_csv.sample(frac=1).reset_index(drop=True)
    # listing target values based on which we will set up the stratified folds
    y = train_csv.target.values
    # initiating the k-fold class from model selection
    kf = model_selection.StratifiedKFold(n_splits=5)

    for count, (train_index,val_index) in enumerate(kf.split(X = train_csv,y = y)):
        train_csv.loc[val_index,'kfold'] = count

    train_csv.to_csv(root_dir + "train_folds.csv")


# data loading and converting into tensor
# train data
train_csv = pd.read_csv(root_dir + 'train.csv')
images_id_train = train_csv.image_name.values.tolist()
train_images = [os.path.join(root_dir + 'train/' ,i +'.jpg') for i in images_id_train]
train_targets = train_csv.target.values

# test data
test_csv = pd.read_csv(root_dir + 'test.csv')
images_id_test = test_csv.image_name.values.tolist()
test_images = [os.path.join(root_dir + 'test/' ,i +'.jpg') for i in images_id_test]
test_targets = test_csv.target.values


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
# adding a simple augmentation
aug = albumentations.Compose([
    albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
])

# Loading customDataset and
train_dataset = ClassificationDataset(image_paths=train_images, targets=train_targets, resize=(224,224), augmentations=aug)
test_dataset = ClassificationDataset(image_paths=test_images, targets= test_targets, resize=(224,224), augmentations=aug)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=4)

# defining model
model = models.resnext50_32x4d(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(2048,1000),
    nn.Dropout(p=0.5),
    nn.Linear(1000,1)
)
