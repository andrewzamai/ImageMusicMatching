"""
Multiclass image classification used to detect which objects are contained in an image

1) Import 5 pretrained models

2) Import image and preprocess it for each model input requirements

3) Predict using each different model (multiclass pretrained models softmax output layer)

4) Take for each one the 5 bests

5) Obtain dict of <= 25 labels with probabily (SUM probability if equal label)

6) Return argmax label or 3 best ones

"""

import numpy as np
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_inputResNet50V2, decode_predictions 

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as preprocess_inputXception

from tensorflow.keras.applications.efficientnet import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_inputEfficientNetB2

from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as preprocess_inputEfficientNetV2L

from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_DenseNet201


resNet50V2Model = ResNet50V2(weights='imagenet')
xceptionModel = Xception(weights='imagenet')
efficientNetB2Model = EfficientNetB2(weights='imagenet')
efficientNetV2LModel = EfficientNetV2L(weights='imagenet')
denseNet201Model = DenseNet201(weights='imagenet')

""" ------------------------------------------------------------------------------------------------------------------------------- """

# Import image and preprocess it

img_path = 'content/chips.jpeg'

img224_224 = image.load_img(img_path, target_size=(224, 224))

img299_299 = image.load_img(img_path, target_size=(299, 299))

img260_260 = image.load_img(img_path, target_size=(260, 260))

img480_480 = image.load_img(img_path, target_size=(480, 480))

#ResNet50V2

x224_224 = image.img_to_array(img224_224)
x224_224 = np.expand_dims(x224_224, axis=0)
x224_224 = preprocess_inputResNet50V2(x224_224)

#Xception

x299_299 = image.img_to_array(img299_299)
x299_299 = np.expand_dims(x299_299, axis=0)
x299_299 = preprocess_inputXception(x299_299)

#EfficientNetB2

x260_260 = image.img_to_array(img260_260)
x260_260 = np.expand_dims(x260_260, axis=0)
x260_260 = preprocess_inputEfficientNetB2(x260_260)

#EfficientNetV2L

x480_480 = image.img_to_array(img480_480)
x480_480 = np.expand_dims(x480_480, axis=0)
x480_480 = preprocess_inputEfficientNetV2L(x480_480)

#DenseNet201

x224_224Dense = image.img_to_array(img224_224)
x224_224Dense = np.expand_dims(x224_224Dense, axis=0)
x224_224Dense = preprocess_input_DenseNet201(x224_224Dense)

""" ------------------------------------------------------------------------------------------------------------------------------- """

resNet50V2Preds = resNet50V2Model.predict(x224_224)
# decode the results into a list of tuples (class, description, probability)(one such list for each sample in the batch)
resNet50V2PredsList = decode_predictions(resNet50V2Preds, top=5)[0]
print('ResNet50V2 prediction:', resNet50V2PredsList)

xceptionPreds = xceptionModel.predict(x299_299)
# decode the results into a list of tuples (class, description, probability)
xceptionPredsList = decode_predictions(xceptionPreds, top=5)[0]
print('Xception prediction:', xceptionPredsList)

efficientNetB2Preds = efficientNetB2Model.predict(x260_260)
# decode the results into a list of tuples (class, description, probability)
efficientNetB2PredsList = decode_predictions(efficientNetB2Preds, top=5)[0]
print('EfficientNetB2 prediction:', efficientNetB2PredsList)

efficientNetV2LPreds = efficientNetV2LModel.predict(x480_480)
# decode the results into a list of tuples (class, description, probability)
efficientNetV2LPredsList = decode_predictions(efficientNetV2LPreds, top=5)[0]
print('DenseNet201 prediction:', efficientNetV2LPredsList)

denseNet201Preds = denseNet201Model.predict(x224_224Dense)
# decode the results into a list of tuples (class, description, probability)
denseNet201PredsList = decode_predictions(denseNet201Preds, top=5)[0]
print('DenseNet201 prediction:', denseNet201PredsList)

ensembleScores = dict()

for i in resNet50V2PredsList:
  ensembleScores[i[1]] = i[2]
  
print(ensembleScores)

for i in xceptionPredsList:
  if ensembleScores.get(i[1], -1) == -1:
    ensembleScores[i[1]] = i[2]
  else:
    ensembleScores[i[1]] = ensembleScores.get(i[1]) + i[2]

for i in efficientNetB2PredsList:
  if ensembleScores.get(i[1], -1) == -1:
    ensembleScores[i[1]] = i[2]
  else:
    ensembleScores[i[1]] = ensembleScores.get(i[1]) + i[2]

for i in efficientNetV2LPredsList:
  if ensembleScores.get(i[1], -1) == -1:
    ensembleScores[i[1]] = i[2]
  else:
    ensembleScores[i[1]] = ensembleScores.get(i[1]) + i[2]

for i in denseNet201PredsList:
  if ensembleScores.get(i[1], -1) == -1:
    ensembleScores[i[1]] = i[2]
  else:
    ensembleScores[i[1]] = ensembleScores.get(i[1]) + i[2]    

print(ensembleScores)


