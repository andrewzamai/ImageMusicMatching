import os
import cv2

# Cleaning lines in *_matching.txt that contains musicIDs or imageIDs that we don't have

pathToImages = '/Users/andrew/Projects/ImageMusicMatching/ImagesDataset/ImagesAll'

pathToMP3Folder = '/Users/andrew/Projects/ImageMusicMatching/MusicDEAMdataset/DEAM_audio/MEMD_audio'

# given *_matching.txt compute how many images are missing in ImagesAll

# *_matching.txt single line structure is: musicClipID [space] imageID [space] matchingScore\n

# load all image names that are stored in ImageAll directory
imagesAllIDList = os.listdir(pathToImages)
# Are they all jpg?
for file in imagesAllIDList:
    if file[-4:] != '.jpg':
        print("Not all files are .jpg " + file)

# list of all mp3 files that we have
mp3AllList = os.listdir(pathToMP3Folder)


# Deleting matching pairs for which we don't have image yet (NAPS dataset missing yet) or we don't have song

missingImgIDList = [] # list of all image names in txt matching file for which do not exist image in ImageAll
missingMusicIDList = [] # list of all music names in txt matching file for which do not exist mp3 

IMEMNet_PairsTxtFileList = os.listdir('/Users/andrew/Projects/ImageMusicMatching/IMEMNet_PairsTxt')
imagesIDsInMatchingTxtFilesList = [] # list of all images IDs founded in txt files
for f in IMEMNet_PairsTxtFileList:
    if f != '.DS_Store' and os.path.isfile('/Users/andrew/Projects/ImageMusicMatching/IMEMNet_PairsTxt' + '/' + f):
        file = open('/Users/andrew/Projects/ImageMusicMatching/IMEMNet_PairsTxt/' + f, 'r')
        for line in file:
            musicIDComplete, imageID, score = line.split(' ')
            musicID, mucicIDframe = musicIDComplete.split('-')

            completeImageID = imageID + '.jpg'
            if completeImageID not in imagesAllIDList:
                missingImgIDList.append(completeImageID)
            
            if (musicID + '.mp3') not in mp3AllList:
                missingMusicIDList.append(musicIDComplete)

            imagesIDsInMatchingTxtFilesList.append(imageID) # adding imageID to known IDs list in txt files

print(missingImgIDList)
print(len(missingImgIDList))

print(missingMusicIDList)



# Until we have NAPS dataset also delete pairs with missingImgIDList in it
for f in IMEMNet_PairsTxtFileList:
    if f != '.DS_Store' and os.path.isfile('/Users/andrew/Projects/ImageMusicMatching/IMEMNet_PairsTxt' + '/' + f):
        file = open('/Users/andrew/Projects/ImageMusicMatching/IMEMNet_PairsTxt/' + f, 'r')
        newFile = open('/Users/andrew/Projects/ImageMusicMatching/IMEMNet_PairsTxt/' + f[:-4] + "_cleaned.txt", 'w')
        for line in file:
            musicIDComplete, imageID, score = line.split(' ')
            musicID, mucicIDframe = musicIDComplete.split('-')
            completeImageID = imageID + '.jpg'

            if completeImageID not in missingImgIDList:
                newFile.write(line)


# Count number of lines in each file train, val, test after cleaning
pairsCount = []
for f in IMEMNet_PairsTxtFileList:
    if f != '.DS_Store' and f[-12:] == '_cleaned.txt' and os.path.isfile('/Users/andrew/Projects/ImageMusicMatching/IMEMNet_PairsTxt' + '/' + f) :
        file = open('/Users/andrew/Projects/ImageMusicMatching/IMEMNet_PairsTxt/' + f, 'r')
        for count, line in enumerate(file):
            pass
        pairsCount.append((count + 1)) # number of lines

print("Number of training pairs: " + str(pairsCount[0]) + ", number of validation pairs: " + str(pairsCount[1]) + ", number of test pairs: " + str(pairsCount[2])) # NOT In right order


'''
# cleaning Images: imagesIDsInMatchingTxtFilesList was filled above
numberOfUselessImages = 0
for img in imagesAllIDList:
    if img not in imagesIDsInMatchingTxtFilesList:
        #print(img)
        numberOfUselessImages = numberOfUselessImages + 1
        if os.path.exists('/Users/andrew/Projects/ImageMusicMatching/ImagesDataset/ImagesAll/' + img):
            os.remove('/Users/andrew/Projects/ImageMusicMatching/ImagesDataset/ImagesAll/' + img)

print(numberOfUselessImages)

# We have 5289 images in ImagesAll that are not used in any mathing pair in ImageAll

imagesAllIDListNew = os.listdir('/Users/andrew/Projects/ImageMusicMatching/ImagesDataset/ImagesAll')
numberOfUselessImagesNow = 0
for img in imagesAllIDListNew:
    if img not in imagesIDsInMatchingTxtFilesList:
        numberOfUselessImagesNow = numberOfUselessImagesNow + 1
print(numberOfUselessImagesNow)


'''

