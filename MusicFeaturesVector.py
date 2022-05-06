import pandas as pd
from lzma import MF_BT2
from nbformat import write
import numpy as np
import os
from pyexpat import features
from pydub import AudioSegment
import random

from python_speech_features import mfcc
import scipy.io.wavfile as wav


class MusicFeaturesVector:

    #folder where temporary wav file will be stored
    pathToWavFolder = '/Users/andrew/Projects/ImageMusicMatching/MusicDEAMdataset/DEAM_audio/WAV_audio' 

    def __init__(self, pathToMp3Folder):
        self.pathToMp3Folder = pathToMp3Folder

    # fileMp3Name assumed to have .mp3 extension
    def convertFromMp3ToWav(self, fileMp3Name):
        sound = AudioSegment.from_mp3(self.pathToMp3Folder + '/' + fileMp3Name)
        sound.export(self.pathToWavFolder + '/' + fileMp3Name[:-4] + '.wav', format="wav")

    def getMusicFeaturesVector(self, fileMp3Name):

        #self.convertFromMp3ToWav(fileMp3Name)

        (rate, sig) = wav.read(self.pathToWavFolder + '/' + fileMp3Name[:-4] + '.wav')
        mfcc_feat = mfcc(sig, rate)
        # take only 40 casual frames features 
        index = int(random.uniform(1000, mfcc_feat.shape[0]-1000))
        reshaped_mfcc = np.reshape(mfcc_feat[index : index+40, :], (1, 520))[0, :512]

        #delete wav file
        #os.remove(self.pathToWavFolder + '/' + fileMp3Name[:-4] + '.wav')

        return reshaped_mfcc


    def getMusicFeaturesVector4MyMusic(self, fileMp3Name):

        self.convertFromMp3ToWav(fileMp3Name)

        (rate, sig) = wav.read(self.pathToWavFolder + '/' + fileMp3Name[:-4] + '.wav')
        mfcc_feat = mfcc(sig, rate)
        # take only 40 casual frames features 
        index = int(random.uniform(2000, mfcc_feat.shape[0]-1000))
        reshaped_mfcc = np.reshape(mfcc_feat[index : index+40, :], (1, 520))[0, :512]

        #delete wav file
        os.remove(self.pathToWavFolder + '/' + fileMp3Name[:-4] + '.wav')

        return reshaped_mfcc


'''
existingMp3Files = []
musicFeaturesVector = MusicFeaturesVector('/Users/andrew/Projects/ImageMusicMatching/MusicDEAMdataset/DEAM_audio/MEMD_audio')

for file in os.listdir(musicFeaturesVector.pathToMp3Folder):
    if os.path.exists(musicFeaturesVector.pathToMp3Folder + '/' + file):
        musicFeaturesVector.convertFromMp3ToWav(file)
        existingMp3Files.append(file[:-4])

missingFilesList = open('/Users/andrew/Projects/ImageMusicMatching/MusicDEAMdataset/DEAM_audio/missingMp3Files.txt', 'w')
for i in range(2049):
    if i not in existingMp3Files:
        missingFilesList.write(str(i) + ',')
'''
        
    
    




'''
# Test
musicFeaturesVector = MusicFeaturesVector('/Users/andrew/Projects/ImageMusicMatching/MusicDEAMdataset/DEAM_audio/MEMD_audio')
mfv = musicFeaturesVector.getMusicFeaturesVector('101.mp3').reshape((1,512))

print(mfv)
'''


'''
# To read music.csv file in pandas Dataframe and convert it to numpy
musicFeaturesCSVFolderPath = '/Users/andrew/Projects/ImageMusicMatching/MusicDEAMdataset/features/features'
fileNumber = 5 # which file to read
df = pd.read_csv(musicFeaturesCSVFolderPath + '/' + str(fileNumber) + '.csv', delimiter = ';', header=0)

file_np = pd.DataFrame(df).to_numpy()

'''

