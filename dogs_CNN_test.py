import numpy
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine.saving import model_from_json
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
import os
import csv
import random
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical


#constants
CHIHUAHUA = 0
PUG = 1
breed_name = ["CHIHUAHUA", "PUG"]
MODEL = "TRANSFER_LEARNING" #available models["SIMPLE_CNN", "TRANSFER_LEARNING"]
FINE_TUNING = True
IMG_SIZE = 350
CHANNELS = 3
num_of_images = [152, 200]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
path="breeds"
categories = os.listdir(path)
img_names = [os.listdir(path + "/" + categories[CHIHUAHUA]), os.listdir(path + "/" + categories[PUG])]


def load_model():
    suffix=""
    if FINE_TUNING:
        suffix = "_fine_tuned"
    json_file = open("model"+suffix+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("weights"+suffix+".h5")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model

def loop_over_test_set():
    imgs_list=[]
    dog_breed=[]
    with open("dataset.csv",'r') as file:
        reader = csv.reader(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        row_ind = 0
        for riga in reader:
            if(row_ind == 0):
                row_ind += 1 #skip header
                continue
            file_name = riga[1]
            curr_img = cv2.imread(path + "//" + file_name)
            curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
            array_imm = Image.fromarray(curr_img, 'RGB')
            resized_img = array_imm.resize((IMG_SIZE, IMG_SIZE))
            imgs_list.append(np.array(resized_img))
            dog_breed.append(str(riga[2]))
    file.close()
    imm = np.array(imgs_list)
    labels = np.array(dog_breed)
    random_seed = 42
    rif = np.arange(imm.shape[0])
    np.random.seed(random_seed)
    np.random.shuffle(rif)
    imm = imm[rif]
    labels = labels[rif]
    imm = imm.astype(np.float32)
    labels = labels.astype(np.int32)
    imm = imm/255 #normalisation
    _, xTest, _, yTest = train_test_split(imm,labels,test_size=0.2, random_state=random_seed)
    print("\nx_test shape = ",xTest.shape)
    print("y_test shape = ",yTest.shape)
    y_test_cat = to_categorical(yTest)
    loaded_model_from_disk = load_model()
    loss, accuracy = loaded_model_from_disk.evaluate(xTest, y_test_cat)
    print("loss: ", round(loss,2))
    print("accuracy: ", round(accuracy,2))
    y_test_cat = y_test_cat.astype(int)
    real_class = np.argmax(y_test_cat, axis=1)
    pred_class = loaded_model_from_disk.predict_classes(xTest)
    conf = np.max(loaded_model_from_disk.predict_proba(xTest), axis=1)
    report = classification_report(real_class, pred_class)
    print("classification report:\n" + str(report) + "\n")
    cm = confusion_matrix(real_class, pred_class)
    print("confusion_matrix:\n" + str(cm) + "\n")
    #loop over test set images
    for i in range(len(xTest)):
        plt.figure(1,figsize=(8,4))
        image = xTest[i]
        plt.imshow(image)
        plt.title('Actual breed = {}, Prediction = {}, Confidence = {}'.format(breed_name[real_class[i]],breed_name[pred_class[i]],str(round(conf[i],2))))
        plt.xticks([])
        plt.yticks([])
        plt.show()


def predict(folder_path):
    loaded_model_from_disk = load_model()
    imgsToPredict = []
    imgsToPlot = []
    for img_name in os.listdir(folder_path):
        curr_img = cv2.imread(folder_path+"//"+img_name)
        curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
        array_imm = Image.fromarray(curr_img, 'RGB')
        imgsToPlot.append(array_imm)
        resized_img = array_imm.resize((IMG_SIZE, IMG_SIZE))
        imm = np.array(resized_img, dtype=np.float32)
        imm = imm / 255  # normalisation
        imgsToPredict.append(imm)
    imgsToPredict = np.array(imgsToPredict)
    pred_class = loaded_model_from_disk.predict_classes(imgsToPredict)
    conf = np.max(loaded_model_from_disk.predict_proba(imgsToPredict), axis=1)
    for i in range(len(imgsToPredict)):
        plt.figure(1, figsize=(6, 4))
        plt.imshow(imgsToPlot[i])
        plt.title('Predicted breed = {}, Confidence = {}'.format(breed_name[pred_class[i]], str(round(conf[i],2))))
        plt.xticks([])
        plt.yticks([])
        plt.show()
    return


predict("Emy_imgs")
#loop_over_test_set()




