''' --------------------------------------------- IMAGE MUSIC MATCHING APP: find the perfect song for your photo --------------------------------------------- '''

import sys
import pandas as pd
import numpy as np
import random
#print(sys.path)
''' 
loadImageAndDetect is a module written by us that run an external executable "./darknet" and collects its output that would otherwise be printed at console.
We weren't able to execute that exe in other positions so loadImageAndDetect.py is in that same folder + does some other preprocessing
We now specify to sys where to find module
'''
sys.path.append('/Users/andrew/Projects/ImageMusicMatching/srcCode/ImageDetectionStage/darknet') 
import loadImageAndDetect

sys.path.append('/Users/andrew/Projects/ImageMusicMatching/srcCode/ModelSrcCode')
from ImageFeaturesVector import ImageFeaturesVector
from MusicFeaturesVector import MusicFeaturesVector

sys.path.append('/Users/andrew/Projects/ImageMusicMatching/srcCode/words_extraction')
import filter_song

from tensorflow import keras # to load model saved as h5
from pickle import dump # to load scaler
from pickle import load 

from googletrans import Translator 


# 1) Loading image and detecting which objects are present in it
pathToImage = '/Users/andrew/Projects/ImageMusicMatching/NewMusicImages/Images/rainyDay.jpg' # will be converted to jpg for object detection, for feature extraction?

detectedObjects = loadImageAndDetect.detectObjectsInImage(pathToImage) # Use YOLO net to detect which objects are in the image and return them as list

print("\nList of detected objects in the image: " + '\n')
print(detectedObjects) # list of detected objects in the image (in ENGLISH)
print('\n\n')

# TRANSLATE TO ITALIAN 
translator = Translator()
detectedObjects_new = []
for i in range(len(detectedObjects)):
    translatedObject = translator.translate(detectedObjects[i], dest='it').text.lower() 
    # if more than 2 words split and added separately
    for item in translatedObject.split():
        detectedObjects_new.append(item)

print("List of detected objects in the image, in Italian: " + '\n')
print(detectedObjects_new) # list of detected objects in the image, now in Italian
print('\n')

# 2) I have now a list of names of detected objects in the image; we now proceed in filtering the songs keeping only those that have #words > Threshold

pathToSingleFileAllLyrics = '/Users/andrew/Projects/ImageMusicMatching/srcCode/words_extraction/words_file.txt'
#detectedObjectsList = ['bottiglia', 'crack', 'elefante', 'pizza', 'ciao']
filteredSongsListofTuples = filter_song.getFilteredMusicList(detectedObjects_new, pathToSingleFileAllLyrics) # list of tuples

print("List of filtered songs (songName > object, numberOfOccurrences): " + '\n')
print(filteredSongsListofTuples)
print('\n')

# Since the list of filtered songs is returned in decreasing order of occurrences I could set a threshold on number of songs to match with
# Not done for now.

# construct a list of only song names 
filteredSongsList = []
songNameWordDict = {}
for item in filteredSongsListofTuples:
    if item[0] != '': # why returning some empty tuples?
        songName, object = item[0].split('>')
        filteredSongsList.append(songName)
        songNameWordDict[songName] = (object, item[1]) # using songName as key to save tuple (word, numberOccurrences)

print(songNameWordDict)
#print(filteredSongsList)

# 3) A list of songs is returned I now need to compute all possible image-song feature vectors pairs
# we get feature vector for the image using ImageFeaturesVector.py
# we get feature vector for each song using MusicFeaturesVector.py (use function getMusicFeaturesVector4MyMusic())
# to speed up process we could store al features vectors of all songs in some file
# hstack the vectors and feed to trained model to get matching score --> do for all songs and return top 5 matching

# Load precomputed music features from csv
whereToFindCSV = '/Users/andrew/Projects/ImageMusicMatching/NewMusicImages/NewMusicAllDatabase'
musicFeaturesVector_df = pd.read_csv(whereToFindCSV + '/allMusicFeaturesVectors.csv', sep=',', header=None)
songNames = musicFeaturesVector_df.iloc[:, 0] # object of type Series
songFeatures = musicFeaturesVector_df.iloc[:, 1:].to_numpy() # now I can access a song features vector using it's index 

# Assume we get as filtered songs the list below (left for debugging)
#filteredSongsList = ['Madame+MAREA', 'MondoMarcio+Tieniduro!', 'sangiovanni+farfalle', 'Ernia+Superclassico', 'BLANCO+BluCeleste']

# compute image feature vector
imageFeaturesVector = ImageFeaturesVector()
imgFeatVec = imageFeaturesVector.createImageFeaturesVector(pathToImage).reshape((1,512))


# Problem: if empty list? get 5/10 random songs?
noSongsWereDetected = False
if len(filteredSongsList) == 0:
    print("\n No songs were found for detected objects, finding 10 in a random way!")
    noSongsWereDetected = True
    for i in range(10):
        randomValue = random.randrange(0, len(songNames))
        filteredSongsList.append(songNames[randomValue])

    print(filteredSongsList)

index = songNames[songNames == filteredSongsList[0]].index[0] # retrieve row number given song name
musicFeatVec = songFeatures[index].reshape((1,512)) # get music features vector at index and reshape
imgMusFeatVec = np.hstack((imgFeatVec, musicFeatVec)) # concatenate the 2 features vectors
featuresMatrix = imgMusFeatVec # will be first row of matrix where each row an instance and each column a feature

# compute for all image-songs pairs
for i in range(1, len(filteredSongsList)):
    index = songNames[songNames == filteredSongsList[i]].index[0]
    musicFeatVec = songFeatures[index].reshape((1,512))
    imgMusFeatVec = np.hstack((imgFeatVec, musicFeatVec))
    featuresMatrix = np.vstack((featuresMatrix, imgMusFeatVec)) # append to featuresMatrix

#print(featuresMatrix)

# Train/retrieve model weigths and scaler
#model, scaler = model.trainModel()

# instead of training we load pretrained model and scaler
model = keras.models.load_model('/Users/andrew/Projects/ImageMusicMatching/srcCode/ModelSrcCode/imageMusicTrainedModel.h5', compile=False)
scaler = load(open('/Users/andrew/Projects/ImageMusicMatching/srcCode/ModelSrcCode/scaler.pkl', 'rb'))

featuresMatrix_scaled = scaler.transform(featuresMatrix) # scale features using scaler fitted on X_train when model was trained

predictions = {} # dictionary that will store for each song name its matching score with the loaded image

for i in range(len(featuresMatrix_scaled)):
    prediction = model.predict(featuresMatrix_scaled[i].reshape((1,1024))) 
    predictions[filteredSongsList[i]] = prediction[0,0]
print('\n')

print("Here is the list of suggested songs: \n")
finalSongsSorted = sorted(predictions.items(), key=lambda x: x[1], reverse=True) # returning a sorted list by decreasing value no more dictionary as was predictions
if noSongsWereDetected == False:
    for item in finalSongsSorted:
        print("--> Song: " + item[0] + " with a score on sentiment of: " + str(item[1]) + ", because the following object: " + songNameWordDict.get(item[0])[0] + " occurred " + str(songNameWordDict.get(item[0])[1]) + " times. \n")
else:
    print(predictions)

