import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine.saving import model_from_json
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import os
import csv
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam
from tensorflow.keras.applications import vgg16

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


def build_simple_CNN():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3),strides=(2,2), padding="valid",
                     activation="relu", input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS)))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(2,2), padding="valid",
                     activation="relu"))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation="softmax"))
    model.summary()
    return model


def transfer_learning():
    vgg_conv = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS))
    for layer in vgg_conv.layers[:]:
        layer.trainable = False
    model = Sequential()
    model.add(vgg_conv)
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.summary()
    return model


def load_model():
    json_file = open("model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("weights.h5")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model


######### DOG BREED RECOGNITION IN TEN STEPS #########


#1) create cvs dataset
with open('dataset.csv','w',newline='') as file:
    csvwriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['ID', 'File_name', 'Breed']) #header
    for id in range(num_of_images[CHIHUAHUA]+num_of_images[PUG]):
        breed = CHIHUAHUA
        offset = 0
        if id >= num_of_images[CHIHUAHUA]:
            breed = PUG
            offset = num_of_images[CHIHUAHUA]
        csvLine = []
        csvLine.append(id)
        csvLine.append(categories[breed] + "\\" + img_names[breed][id - offset])
        csvLine.append(breed)
        csvwriter.writerow(csvLine)
file.close()


#2) read dataset and load images
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


#3) plot some test images
plt.figure(1, figsize = (12,5))
n=0
for i in range(8):
    n+=1
    r = np.random.randint(0,imm.shape[0],1)
    plt.subplot(2,4,n)
    plt.imshow(imm[r[0]])
    plt.title('Dog breed : {}'.format(breed_name[int(labels[r[0]])]))
    plt.xticks([])
    plt.yticks([])
plt.show()
print(imm[0].shape)


#4) prepare train and test sets
random_seed = 42
rif = np.arange(imm.shape[0])
np.random.seed(random_seed)
np.random.shuffle(rif)
imm = imm[rif]
labels = labels[rif]
imm = imm.astype(np.float32)
labels = labels.astype(np.int32)
imm = imm/255 #normalisation
xTrain, xTest, yTrain, yTest = train_test_split(imm,labels,test_size=0.2, random_state=random_seed)
print("x_train shape = ",xTrain.shape)
print("y_train shape = ",yTrain.shape)
print("\nx_test shape = ",xTest.shape)
print("y_test shape = ",yTest.shape)
y_train_cat = to_categorical(yTrain)
y_test_cat = to_categorical(yTest)


#5) create generator for data augmentation
datagenTrain = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
datagenTrain.fit(xTrain)


#6) create the model
prefix = ""
if MODEL == "SIMPLE_CNN":
    prefix = "simpleCNN"
    model = build_simple_CNN()
elif MODEL == "TRANSFER_LEARNING":
    model = transfer_learning()
else:
    print("Model not available, using default TRANSFER_LEARNING")
    model = transfer_learning()


#7) model compiling and training
# compile model
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
# set callbacks for early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
best_weights_file = prefix+"weights.h5"
mc = ModelCheckpoint(best_weights_file, monitor='val_loss', mode='min', verbose=2,
                     save_best_only=True)
# train model testing it on each epoch with real time data augmentation
history = model.fit(datagenTrain.flow(xTrain, y_train_cat, save_to_dir= "data_aug", batch_size=32), validation_data=(xTest, y_test_cat),
                    batch_size=32, callbacks= [es, mc], epochs=50, verbose=2)


#8) model testing
# accuracy on validation
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
if MODEL == "SIMPLE_CNN":
    plt.title('model accuracy - simple CNN')
else:
    plt.title('model accuracy - transfer learning')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(prefix+"accuracy.png")
plt.gcf().clear()  # clear
# loss on validation
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
if MODEL == "SIMPLE_CNN":
    plt.title('model loss - simple CNN')
else:
    plt.title('model loss - transfer learning')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(prefix+"loss.png")
plt.gcf().clear()  # clear
# test acc and loss
model.load_weights(best_weights_file) # load the best saved model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
test_metrics = model.evaluate(xTest,y_test_cat, batch_size=32)
print("\n%s: %.2f%%" % ("test " + model.metrics_names[1], test_metrics[1] * 100))
print("%s: %.2f" % ("test " + model.metrics_names[0], test_metrics[0]))
print("test accuracy: " + str(format(test_metrics[1], '.3f')) + "\n")
print("test loss: " + str(format(test_metrics[0], '.3f')) + "\n")
# test acc and loss per class
y_test_cat = y_test_cat.astype(int)
real_class = np.argmax(y_test_cat, axis=1)
pred_class = model.predict_classes(xTest)
report = classification_report(real_class, pred_class)
print("classification report:\n" + str(report) + "\n")
cm = confusion_matrix(real_class, pred_class)
print("confusion_matrix:\n" + str(cm) + "\n")
# save neural network on disk:
# serialize model to JSON
model_json = model.to_json()
with open(prefix+"model.json","w") as json_file:
        json_file.write(model_json)
print("model saved")
print()


#9) fine tune the transfer learning model (if fine tuning is enabled)
if FINE_TUNING and model == "TRANSFER_LEARNING":
    model.trainable = True
    best_weights_file = "weights_fine_tuned.h5"
    mc = ModelCheckpoint(best_weights_file, monitor='val_loss', mode='min', verbose=2,
                         save_best_only=True)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5), metrics=['accuracy'])
    history = model.fit(datagenTrain.flow(xTrain, y_train_cat, batch_size=32), validation_data=(xTest, y_test_cat),
                        batch_size=32, callbacks= [es, mc], epochs=50, verbose=2)
    #10) evaluate fine tuned model
    # generate metrics plot:
    # accuracy on validation
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy - fine tuning')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig("accuracy_fineTuning.png")
    plt.gcf().clear()  # clear
    # loss on validation
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss - fine tuning')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig("loss_fineTuning.png")
    plt.gcf().clear()  # clear
    # test acc and loss
    # load the best saved model
    model.load_weights(best_weights_file)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    test_metrics = model.evaluate(xTest,y_test_cat, batch_size=32)
    print("\n%s: %.2f%%" % ("test " + model.metrics_names[1], test_metrics[1] * 100))
    print("%s: %.2f" % ("test " + model.metrics_names[0], test_metrics[0]))
    print("test accuracy: " + str(format(test_metrics[1], '.3f')) + "\n")
    print("test loss: " + str(format(test_metrics[0], '.3f')) + "\n")
    # test acc and loss per class
    y_test_cat = y_test_cat.astype(int)
    real_class = np.argmax(y_test_cat, axis=1)
    pred_class = model.predict_classes(xTest)
    report = classification_report(real_class, pred_class)
    print("classification report:\n" + str(report) + "\n")
    cm = confusion_matrix(real_class, pred_class)
    print("confusion_matrix:\n" + str(cm) + "\n")
    # save neural network on disk:
    # serialize model to JSON
    model_json = model.to_json()
    with open("model_fine_tuned.json","w") as json_file:
            json_file.write(model_json)
    print("model saved")
    print()
