from __future__ import print_function
import warnings
from os import environ
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, Input, Flatten, Activation, Dropout, Dense
from keras.optimizers import Adam, SGD
from keras.initializers import glorot_normal
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.regularizers import l2
from keras import backend as K
from keras.models import Sequential, Model
from keras.preprocessing import image
from os import path, getcwd
from imageAugmentation import *
from dataLoad import *
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
import random
import numpy as np
import shutil

# Model mreže
# (model mreže inspiriran glavnim člankom)
def initialize():
    """Konstruira i inicijalizira osnovnu konvolucijsku neuronsku mrežu"""
    model = Sequential()
    model.add(Conv2D(32,
                    kernel_size=3, 
                    strides=2, 
                    padding='same', 
                    input_shape=input_shape,
                    kernel_initializer=glorot_normal()))

    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(32, 
                    kernel_size=3, 
                    strides=2, 
                    padding="same", 
                    input_shape=input_shape,
                    kernel_initializer=glorot_normal()))

    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(4*num_artists, 
                    input_shape=(6272,),
                    kernel_initializer=glorot_normal()))
    model.add(Activation('relu'))
    model.add(Dense(num_artists, 
                    input_shape=(4*num_artists,),
                    activation='softmax',
                    kernel_initializer=glorot_normal()))

    tbCallBack = TensorBoard(log_dir='./GraphSiroviVgg16-3', 
                            write_graph=True, 
                            write_images=True,
                            write_grads=False)

    mdCheckPoint = ModelCheckpoint(filepath='sirovi-vgg16.weights.{epoch:02d}--{val_acc:.5f}.hdf5',
                                monitor='val_acc',
                                mode='max',
                                save_best_only=False,
                                save_weights_only=False,
                                verbose=1,
                                period=1)

    # koristimo adamov optimizator i metrika je točnost
    model.compile(loss='categorical_crossentropy',
                optimizer=Adam(lr=1e-4),
                metrics=['accuracy'])

    # crta tablicu slojeva mreže
    model.summary()
    return model

def train(model, pretrained):
    if pretrained == True:
    # mrežu smo već istrenirali i spremili njene težine
        model.load_weights('pokusaj300-20.h5')        

    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size

    # treniramo mrežu....
    model.fit_generator(train_crops,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        epochs=10,
                        validation_data=val_crops,
                        validation_steps=STEP_SIZE_VALID,
                        workers=4,
                        callbacks=[
                            TensorBoard(log_dir='./GraphBaseline', 
                                        histogram_freq=1, 
                                        write_graph=True, 
                                        write_images=True,
                                        write_grads=False),
                            ModelCheckpoint(filepath='baseline.weights.{epoch:02d}--{val_acc:.5f}.hdf5',
                                        monitor='val_acc',
                                        mode='max',
                                        save_best_only=False,
                                        save_weights_only=False,
                                        verbose=1,
                                        period=1)])

    model.save_weights('prvi_pokusaj300SVE-SLIKE20.h5')

def evaluate(model):
    evaluation = model.evaluate_generator(test_crops,
                            steps=test_generator.n//test_generator.batch_size,
                            workers=4,
                            verbose=1)

    predictions = model.predict_generator(test_crops, 
                                        steps=test_generator.n//test_generator.batch_size,
                                        workers=4,
                                        verbose=1)

    preds = np.argmax(predictions, axis=-1) # multiple categories

    label_map = (train_generator.class_indices)
    label_map = dict((v,k) for k,v in label_map.items()) # flip k,v
    preds_names = [label_map[k] for k in preds]

    print(sum(preds == test_generator.classes)/len(preds))

    np.save(open('predictions_base_test.npy', 'wb'), predictions)

    img_path = 0
    img = image.load_img("/home/ivana/repos/Vje-ba/testPicasso.jpg")
    x = image.img_to_array(img)

    img_cropped = center_crop(x, (224, 224))
    img_cropped_blah = image.array_to_img(img_cropped)
    img_cropped_blah.show()

    x_pred = model.predict_classes(img_cropped.reshape(1, 224, 224, 3), batch_size=1)
    print('Sliku je naslikao: ', label_map[x_pred[0]])
+
    print(confusion_matrix(test_generator.classes, preds))
    print(classification_report(test_generator.classes, preds))
