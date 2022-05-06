from distutils import extension 
import subprocess # to execute ./darknet detector and get output that would be directed in stdout
from PIL import Image 

# 1) convert image to jpg since YOLO detector used requires .jpg, delete created image then

# delete image created by YOLO detector? 


# receives in input path to any format image and returns a list of detected objects in it
# detector also saves an image with detected image bounded 
def detectObjectsInImage(pathToImage):
    
    # converting to jpg
    img = Image.open(pathToImage)
    rgb_img = img.convert("RGB")
    # exporting the image
    pathToImgName, pathToImgExtension = pathToImage.split('.')
    rgb_img.save(pathToImgName + '.jpg')

    # running ./darknet executable passing yolov3.cfg and weights
    imageJpgPath = pathToImgName + '.jpg'
    #stringToExecute = '/Users/andrew/Projects/ImageMusicMatching/srcCode/ImageDetectionStage/darknet/darknet detect cfg/yolov3.cfg yolov3.weights ' + imageJpgPath

    stringToExecute = '/Users/andrew/Projects/ImageMusicMatching/srcCode/ImageDetectionStage/darknet/darknet detect cfg/yolov3.cfg yolov3.weights ' + imageJpgPath
    # The called detector will also save an image with bounding boxes on detected objects

    # Getting output of the program and decoding it, something like: b'/Users/andrew/Desktop/Tests/chips.jpg: Predicted in 18.827434 seconds.\nbottle: 99%\nbottle: 79%\n'
    proc = subprocess.Popen(stringToExecute, stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    # splitting based on space and \n \t: 0th element path to img, 1st to 4th Predicted in 19 seconds., then detected object followed by percentage in next slot
    out_splitted = out.split() 

    print(out_splitted)

    detectedObjects = []
    lastWasString = False
    for i in range(5, len(out_splitted), 1): # skip percentages in between, attention to words of 2 items like teddy bear
        # if out_splitted[i+1] > 30%
        if lastWasString == True:
            lastWasString = False
            continue

        object, lastLetter = out_splitted[i].decode('utf-8')[:-1], out_splitted[i].decode('utf-8')[-1]
        try:
            string_int = int(object) 
            object = string_int   
        except ValueError:
            pass
        if isinstance(object, str) and i<len(out_splitted)-1:
            nextItem = out_splitted[i+1].decode('utf-8')[:-1]
            try:
                string_int = int(nextItem) 
                nextItem = string_int   
            except ValueError:
                pass
            if isinstance(nextItem, str):
                object = object + lastLetter + ' ' + nextItem
                lastWasString = True
  
            detectedObjects.append(object)


    return detectedObjects


#print(detectObjectsInImage('/Users/andrew/Projects/ImageMusicMatching/NewMusicImages/Images/pennyAndrew.jpg'))
