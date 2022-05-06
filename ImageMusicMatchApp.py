# 1) load image and compute feature vector

# 2) load music and compute feature vector

# 3) retrieve model and predict

from ImageFeaturesVector import ImageFeaturesVector
from MusicFeaturesVector import MusicFeaturesVector

import model

import pandas as pd
import numpy as np
import csv
import os

from PIL import Image

pathToPhotos4Testing = '/Users/andrew/Projects/ImageMusicMatching/ImagesDataset/Photos4Testing'

imageFeaturesVector = ImageFeaturesVector()

#im1 = Image.open('/Users/andrew/Projects/ImageMusicMatching/ImagesDataset/Photos4Testing/IMG_0181.HEIC', 'r')
#im1.save('/Users/andrew/Projects/ImageMusicMatching/ImagesDataset/Photos4Testing/laureaAndrew.jpg')

imgFeatVec = imageFeaturesVector.createImageFeaturesVector('/Users/andrew/Projects/ImageMusicMatching/ImagesDataset/Photos4Testing/grigliata.jpg')
print(imgFeatVec)

pathToMusic4Testing = '/Users/andrew/Projects/ImageMusicMatching/MusicDEAMdataset/Music4Testing'
musicFeaturesVector = MusicFeaturesVector(pathToMusic4Testing)

musicFeatVec = musicFeaturesVector.getMusicFeaturesVector4MyMusic('EstateEStoQua.mp3').reshape((1,512))

print(musicFeatVec)

imgMusFeatVec = np.hstack((imgFeatVec, musicFeatVec))

print(imgMusFeatVec)


model, scaler = model.trainModel()

imgMusFeatVec_scaled = scaler.transform(imgMusFeatVec)
prediction = model.predict(imgMusFeatVec_scaled)

print(prediction)


