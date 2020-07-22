import pandas as pd
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import os
from sklearn import model_selection
from CustomDataset import ClassificationDataset

# Folding the validation set using stratified cross validation . Stratified cross validation works well for skewed dataset
if __name__ == "__main__":
    train_csv = pd.read_csv('train.csv')
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

    train_csv.to_csv("train_folds.csv")


# data loading and converting into tensor

train_folds = pd.read_csv('train_folds.csv')
root_dir = 'D:/Deep Learning/Projects/Melanoma skin cancer detection/melanoma/train/'
images_id = train_folds.image_name.values.tolist()
images = [os.path.join(root_dir,i +'.jpg') for i in images_id]
targets = train_csv.target.values
dataset = ClassificationDataset(image_paths=images, targets=targets, resize=None, augmentations=None)

for img,label in dataset:
    print(img.shape)
    break
