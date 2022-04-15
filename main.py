'''
Image-Music matching main

TODO: clean more matching pairs with not present songs IDs: DONE

1) Datasets loading: we will use the matching txt files to create our numpy matrices
    eg. lines i of train_matching_cleaned.txt will create sample X_train[i] where its obtained as concatenation of feature vector extracted from pretrained CNN (+else see Building emotional Machines paper) and music feature vector 
    target is third field of line
    
    store them in csv format so to not be every time computed and easily uploaded into numpy matrix

2) define model, early stopping, monitor loss curves (define function)

'''


import pandas as pd
import numpy as np
import csv
import os


pathToFeaturesCSVFolder = '/Users/andrew/Projects/ImageMusicMatching/IMEMNet_PairsTxt/FeaturesCSV'

df = pd.read_csv(pathToFeaturesCSVFolder + '/validCSV.csv', sep=',', header=None)
#print(df.values)       

valid_np = pd.DataFrame(df).to_numpy()
X_valid_np, Y_valid_np = valid_np[:, :-1], valid_np[:, -1]

print(X_valid_np)
print(Y_valid_np)



# Normalize Features


# Build Model




'''
pathToImages = '/Users/andrew/Projects/ImageMusicMatching/ImagesDataset/ImagesAll/'
imageName = 'COCO_train2014_000000567630'
imageFeatureVector = ImageFeaturesVector()
print((imageFeatureVector.createImageFeaturesVector(pathToImages + imageName + ".jpg")))


'''





