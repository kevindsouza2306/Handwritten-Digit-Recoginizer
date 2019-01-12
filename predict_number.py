from keras.models import load_model
import numpy as np
import cv2
from sklearn.preprocessing import LabelBinarizer

from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledataloader import SimpleDataLoader
from imutils import paths


def PredictNumber():

    classlabels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    print("[INFO] sampling imags")

    imagePaths = np.array(list(paths.list_images("media")))

    idxs = np.random.randint(0, len(imagePaths), size=(10,))
    print(idxs)
    imagePaths = imagePaths[[0]]
    sp = SimplePreprocessor(28, 28)
    iap = ImageToArrayPreprocessor()

    sdl = SimpleDataLoader(preprocessor=[sp, iap])
    (data, labels) = sdl.load(imagePaths)
    data = data.astype("float") / 255.0

    print("[INFO] loading pre-trained network...")
    model = load_model("model/lenet_model.h5")

    print("[INFO] predicting....")

    preds = model.predict(data, batch_size=32).argmax(axis=1)



    return classlabels[preds[0]]
