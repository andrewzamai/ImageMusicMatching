import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_VGG16


class ImageFeaturesVector:

    def __init__(self):
        self.vgg16Model = VGG16(weights = 'imagenet', include_top = False, pooling = 'max')

    def createImageFeaturesVector(self, imageName):
        img224_224 = image.load_img(imageName, target_size = (224, 224))
        x224_224 = image.img_to_array(img224_224)
        x224_224 = np.expand_dims(x224_224, axis=0)

        x224_224 = preprocess_input_VGG16(x224_224)

        return self.vgg16Model.predict(x224_224)