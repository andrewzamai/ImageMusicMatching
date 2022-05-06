from ImageFeaturesVector import ImageFeaturesVector
from MusicFeaturesVector import MusicFeaturesVector


import pandas as pd
import numpy as np
import csv
import os

from datetime import datetime

# Load *_matching.txt, compute features vector for each line and store it as .csv file

pathTo_Train_MatchingTxt = '/Users/andrew/Projects/ImageMusicMatching/IMEMNet_PairsTxt/train_matching_cleaned.txt'
pathTo_Valid_MatchingTxt = '/Users/andrew/Projects/ImageMusicMatching/IMEMNet_PairsTxt/val_matching_cleaned.txt'
pathTo_Test_MatchingTxt = '/Users/andrew/Projects/ImageMusicMatching/IMEMNet_PairsTxt/test_matching_cleaned.txt'

imagesAllPath = '/Users/andrew/Projects/ImageMusicMatching/ImagesDataset/ImagesAll'
mp3FolderPath = '/Users/andrew/Projects/ImageMusicMatching/MusicDEAMdataset/DEAM_audio/MEMD_audio'

imageFeaturesVector = ImageFeaturesVector() # to compute image feature vector (contains VGG16 pretrained model)
musicFeaturesVector = MusicFeaturesVector(mp3FolderPath) # to compute music feature vector

# where to store dataset in csv format
pathToFeaturesCSVFolder = '/Users/andrew/Projects/ImageMusicMatching/IMEMNet_PairsTxt/FeaturesCSV'

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Validation conversion started at: ", current_time)

musicFeaturesDict = {} #already computed feature vectors
imageFeaturesDict = {} #already computed image feature vectors


with open(pathTo_Valid_MatchingTxt, 'r') as validFile:

    with open(pathToFeaturesCSVFolder + '/validCSV.csv', 'w', newline='\n') as validFileCSV:

        for line in validFile:

            musicIDComplete, imageID, score = line.split(' ')
            musicID, mucicIDframe = musicIDComplete.split('-')

            if os.path.exists(imagesAllPath + '/' + imageID + '.jpg') and os.path.exists(mp3FolderPath + '/' + musicID + '.mp3'):

                # musicID contains a value [2, 2058] without csv extension; imageID is without .jpg extension; score is a value [0,1]\n

                # take imageID and compute features vector using ImageFeaturesVector
                if np.any(imageFeaturesDict.get(imageID)):
                    imgFeatVec = imageFeaturesDict.get(imageID)
                else:  
                    imgFeatVec = imageFeaturesVector.createImageFeaturesVector(imagesAllPath + '/' + imageID + '.jpg')
                    imageFeaturesDict[imageID] = imgFeatVec
                
                # concatenate with music feature vector
                if np.any(musicFeaturesDict.get(musicID)):
                    musicFeatVec = musicFeaturesDict.get(musicID)
                else:
                    musicFeatVec = musicFeaturesVector.getMusicFeaturesVector(musicID + '.mp3').reshape((1, 512))
                    musicFeaturesDict[musicID] = musicFeatVec
                
        
                imgMusFeatVec = np.hstack((imgFeatVec, musicFeatVec))

                # concatenate with score value
                completeImgFeatVec = np.append(imgMusFeatVec, eval(score[:-1])).reshape((1, 1025))
                
                mywriter = csv.writer(validFileCSV, delimiter = ',')
                mywriter.writerows(completeImgFeatVec)
                

print("Valid done")
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Validation conversion ended at: ", current_time)

# For test

with open(pathTo_Test_MatchingTxt, 'r') as testFile:

    with open(pathToFeaturesCSVFolder + '/testCSV.csv', 'w', newline='\n') as testFileCSV:

        for line in testFile:

            musicIDComplete, imageID, score = line.split(' ')
            musicID, mucicIDframe = musicIDComplete.split('-')

            if os.path.exists(imagesAllPath + '/' + imageID + '.jpg') and os.path.exists(mp3FolderPath + '/' + musicID + '.mp3'):

                # musicID contains a value [2, 2058] without csv extension; imageID is without .jpg extension; score is a value [0,1]\n

                # take imageID and compute features vector using ImageFeaturesVector
                if np.any(imageFeaturesDict.get(imageID)):
                    imgFeatVec = imageFeaturesDict.get(imageID)
                else:  
                    imgFeatVec = imageFeaturesVector.createImageFeaturesVector(imagesAllPath + '/' + imageID + '.jpg')
                    imageFeaturesDict[imageID] = imgFeatVec
                
                # concatenate with music feature vector
                if np.any(musicFeaturesDict.get(musicID)):
                    musicFeatVec = musicFeaturesDict.get(musicID)
                else:
                    musicFeatVec = musicFeaturesVector.getMusicFeaturesVector(musicID + '.mp3').reshape((1, 512))
                    musicFeaturesDict[musicID] = musicFeatVec

                imgMusFeatVec = np.hstack((imgFeatVec, musicFeatVec))

                # concatenate with score value
                completeImgFeatVec = np.append(imgMusFeatVec, eval(score[:-1])).reshape((1, 1025))
                
                mywriter = csv.writer(testFileCSV, delimiter = ',')
                mywriter.writerows(completeImgFeatVec)

print("Test done")
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Test conversion ended at: ", current_time)


with open(pathTo_Train_MatchingTxt, 'r') as trainFile:

    with open(pathToFeaturesCSVFolder + '/trainCSV.csv', 'w', newline='\n') as trainFileCSV:

        for line in trainFile:

            musicIDComplete, imageID, score = line.split(' ')
            musicID, mucicIDframe = musicIDComplete.split('-')

            if os.path.exists(imagesAllPath + '/' + imageID + '.jpg') and os.path.exists(mp3FolderPath + '/' + musicID + '.mp3'):

                # musicID contains a value [2, 2058] without csv extension; imageID is without .jpg extension; score is a value [0,1]\n

                # take imageID and compute features vector using ImageFeaturesVector
                if np.any(imageFeaturesDict.get(imageID)):
                    imgFeatVec = imageFeaturesDict.get(imageID)
                else:  
                    imgFeatVec = imageFeaturesVector.createImageFeaturesVector(imagesAllPath + '/' + imageID + '.jpg')
                    imageFeaturesDict[imageID] = imgFeatVec
                
                # concatenate with music feature vector
                if np.any(musicFeaturesDict.get(musicID)):
                    musicFeatVec = musicFeaturesDict.get(musicID)
                else:
                    musicFeatVec = musicFeaturesVector.getMusicFeaturesVector(musicID + '.mp3').reshape((1, 512))
                    musicFeaturesDict[musicID] = musicFeatVec
                    
                imgMusFeatVec = np.hstack((imgFeatVec, musicFeatVec))

                # concatenate with score value
                completeImgFeatVec = np.append(imgMusFeatVec, eval(score[:-1])).reshape((1, 1025))
                
                mywriter = csv.writer(trainFileCSV, delimiter = ',')
                mywriter.writerows(completeImgFeatVec)


print("Train done") 
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Train conversion ended at: ", current_time)                          


