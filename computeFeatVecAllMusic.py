# Compute featureVector for every song in database and store it in csv file for faster retrieve

from MusicFeaturesVector import MusicFeaturesVector

import pandas as pd
import numpy as np
import csv
import os


pathToAllMusicMp3 = '/Users/andrew/Projects/ImageMusicMatching/NewMusicImages/NewMusicAllDatabase/mp3Files'
musicFeaturesVector = MusicFeaturesVector(pathToAllMusicMp3)

whereToSaveCSV = '/Users/andrew/Projects/ImageMusicMatching/NewMusicImages/NewMusicAllDatabase'


with open(whereToSaveCSV + '/allMusicFeaturesVectors.csv', 'w', newline='\n') as csvFile:

    for filename in os.listdir(pathToAllMusicMp3):

        if filename != '.DS_Store':

            songName = filename[:-4] #remove .mp3 extension
            songFeatureVector = musicFeaturesVector.getMusicFeaturesVector4MyMusic(filename).reshape((1, 512))

            mywriter = csv.writer(csvFile, delimiter = ',')
            csvFile.write(songName + ',')
            mywriter.writerows(songFeatureVector)



'''

musicFeaturesVector_df = pd.read_csv(whereToSaveCSV + '/allMusicFeaturesVectors.csv', sep=',', header=None)
#train_np = pd.DataFrame(train_df).to_numpy()

songNames = musicFeaturesVector_df.iloc[:, 0]

songFeatures = musicFeaturesVector_df.iloc[:, 1:].to_numpy()

print(songNames)

index = songNames[songNames == 'Mahmood+SferaEbbasta+Feid+Dorado'].index[0]

print(songFeatures[index])


'''

print("FINISH")
