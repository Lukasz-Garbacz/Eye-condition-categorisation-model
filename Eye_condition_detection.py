import keras
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications import VGG16
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.optimizers import Adam
from keras.models import Model 
from keras.models import Sequential
from keras.models import load_model
from keras import optimizers 
from keras import layers
from keras import models
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
%matplotlib inline
from keras.models import load_model
import glob
import os, shutil
from PIL import Image
from tensorflow.keras.preprocessing import image


from google.colab import drive
drive.mount('/content/drive')

cd '/content/drive/MyDrive/train'

#pobieramy bazowy, wytrenowany model VGG16 z wagami
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#blokujemy warstwy bazowe
base_model.trainable = False

model = models.Sequential()
model.add(base_model)

#dodajemy nowe warstwy
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(layers.Dense(320, activation='relu'))
model.add(layers.Dense(160, activation='relu'))
model.add(layers.Dense(80, activation='relu'))
model.add(layers.Dense(40, activation='relu'))
model.add(layers.Dense(20, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(4, activation='softmax'))

opt = Adam(lr=0.0003)
model.compile(optimizer=opt, loss='binary_crossentropy', 
              metrics=['accuracy'])

model.summary()

#pobieramy, denormalizujemy i łączymy dane w batche
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2, 
        height_shift_range=0.2, 
        shear_range=0.2, 
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        '/content/drive/My Drive/train1',
        target_size=(224, 224),
        batch_size=10,
        class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
        '/content/drive/My Drive/test1',
        target_size=(224, 224),
        batch_size=10,
        class_mode='categorical')

filepath="model.{epoch:02d}-{val_accuracy:.2f}.hdf5"
callbacks = [keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', period=1)]

#uczenie modelu, zakomentowane by nie uczyć go przy kazdym wywolaniu programu

#history = model.fit(train_generator, verbose=1, 
#  steps_per_epoch=len(train_generator), epochs=100,
#  validation_data=test_generator,
#  validation_steps=len(test_generator), callbacks=callbacks)

classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
model1 = load_model('/content/drive/MyDrive/train/model_acc_0.97.hdf5')

# funkcja wywolujaca model
def predict_from_image_path(image_path, model):
    imag = image.load_img(image_path, target_size=(224, 224))
    imag = np.array(imag)
    imag = imag.astype('float32')
    imag = (255 - image.img_to_array(imag))/255
    test_imag = imag.reshape((1, 224, 224, 3))
    pred = model.predict(test_imag)
    return pred 
    

#do testow pojedynczych zdjec
#img_path = '/content/drive/MyDrive/test/CNV/CNV-172472-8.jpeg'
#print(predict_from_image_path(img_path, model2))

path_eval = "/content/drive/My Drive/evaluate/files_to_evaluate"

#zbierz wszystkie pliki z tych folderow z powrotem do "files_to_evaluate"
files_arr1 = glob.glob("/content/drive/My Drive/evaluate/CNV/*.jpeg")
files_arr2 = glob.glob("/content/drive/My Drive/evaluate/DME/*.jpeg")
files_arr3 = glob.glob("/content/drive/My Drive/evaluate/DRUSEN/*.jpeg")
files_arr4 = glob.glob("/content/drive/My Drive/evaluate/NORMAL/*.jpeg")
files_arr5 = glob.glob("/content/drive/My Drive/evaluate/UNKNOWN/*.jpeg")

for dir1 in files_arr1:
  shutil.move(dir1,path_eval)
for dir2 in files_arr2:
  shutil.move(dir2,path_eval)
for dir3 in files_arr3:
  shutil.move(dir3,path_eval)
for dir4 in files_arr4:
  shutil.move(dir4,path_eval)
for dir5 in files_arr5:
  shutil.move(dir5,path_eval)

#sciezki folderow do ewaluacji
path_cnv = "/content/drive/My Drive/evaluate/CNV"
path_dme = "/content/drive/My Drive/evaluate/DME"
path_drusen = "/content/drive/My Drive/evaluate/DRUSEN"
path_normal = "/content/drive/My Drive/evaluate/NORMAL"
path_unknown = "/content/drive/My Drive/evaluate/UNKNOWN"

#progi pewnosci ponizej ktorego zdjecia nie sa zaliczane jako dana kategoria
thr_CNV = 0.73
thr_DME = 0.83
thr_DRU = 0.3
thr_NOR = 0.6

#wczytaj pliki
files_arr = glob.glob("/content/drive/My Drive/evaluate/files_to_evaluate/*.jpeg")

#wywolaj model i przenies zdjecie do odpowiedniego folderu
for dir in files_arr:
  pred = predict_from_image_path(dir, model1)
  if pred[0][0] > thr_CNV:
    shutil.move(dir,path_cnv)
  elif pred[0][1] > thr_DME:
    shutil.move(dir,path_dme)
  elif pred[0][2] > thr_DRU and pred[0][2] > pred[0][0] and pred[0][2] > pred[0][1] and pred[0][2] > pred[0][3]:
    shutil.move(dir,path_drusen)
  elif pred[0][3] > thr_NOR:
    shutil.move(dir,path_normal)
  else:
    shutil.move(dir,path_unknown)


#drugi stopien klasyfikacji
files_arr2 = glob.glob("/content/drive/My Drive/evaluate/UNKNOWN/*.jpeg")
model2 = load_model('/content/drive/MyDrive/train/model_bin_acc_0.99.hdf5')

for dir in files_arr2:
  pred = predict_from_image_path(dir, model2)
  if pred[0][0] > 0.95:
    shutil.move(dir,path_drusen)
  elif pred[0][1] > 0.97:
    shutil.move(dir,path_normal)