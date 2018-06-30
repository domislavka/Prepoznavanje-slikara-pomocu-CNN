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
from os import path, getcwd
import pandas as pd
import random
import numpy as np
import shutil

from dataLoad import *
from top3_accuracy import *

trainData = loadTrain()
train_generator = trainData[0]
train_crops = trainData[1]

valData = loadVal()
validation_generator = valData[0]
val_crops = valData[1]

testData = loadTest()
test_generator = testData[0]
test_crops = testData[1]

num_artists = train_generator.num_classes

STEP_SIZE_TRAIN = train_generator.n/train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n/validation_generator.batch_size + 1
STEP_SIZE_TEST = test_generator.n//test_generator.batch_size + 1

img_input = Input((224, 224, 3))

def initialize():

    '''Konstruira i inicijalizira vrstu mreže vgg16 sa 5 blokova '''

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_initializer=glorot_normal())(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_initializer=glorot_normal())(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_initializer=glorot_normal())(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_initializer=glorot_normal())(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_initializer=glorot_normal())(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_initializer=glorot_normal())(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_initializer=glorot_normal())(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_initializer=glorot_normal())(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_initializer=glorot_normal())(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_initializer=glorot_normal())(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_initializer=glorot_normal())(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_initializer=glorot_normal())(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_initializer=glorot_normal())(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    base = Model(img_input, x)

    xx = base.output
    xx = Dense(128, activation='relu', kernel_initializer=glorot_normal())(xx)
    xx = GlobalAveragePooling2D()(xx)
    preds_layer = Dense(num_artists, activation='softmax', kernel_initializer=glorot_normal())(xx)

    my_vgg16 = Model(inputs=base.input, outputs=preds_layer)

    model.compile(loss='categorical_crossentropy',
                         optimizer=Adam(lr=1e-4),
                         metrics=['accuracy'])

    model.summary()

    return my_vgg16


def train(model, pretrained):

    if pretrained == True:
    # mrežu smo već istrenirali i spremili njene težine
        model.load_weights('sirovi_vgg16_tezine.h5')
        return model

    model.fit_generator(train_crops,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        epochs=20,
                        validation_data=val_crops,
                        validation_steps=STEP_SIZE_VALID,
                        workers=4,
                        callbacks=[
                            TensorBoard(log_dir='./GraphSiroviVGG16', 
                                        histogram_freq=1, 
                                        write_graph=True, 
                                        write_images=True,
                                        write_grads=False),
                            ModelCheckpoint(filepath='sirovivgg6.weights.{epoch:02d}--{val_acc:.5f}.hdf5',
                                        monitor='val_acc',
                                        mode='max',
                                        save_best_only=False,
                                        save_weights_only=False,
                                        verbose=1,
                                        period=1)])

    # spremimo tezine
    model.save_weights('sirovi_vgg16_tezine.h5')
    # spremimo model
    model.save('sirovi_vgg16.h5')

    return model


def test(model, loadSavedPreds = True):

    '''
    Funkcija za testiranje mreze SiroviVGG16
    '''


    if loadSavedPreds == True:
        predictions = np.load('predictions_siroviVGG16_test.npy')
    
    else:

        predictions = model.predict_generator(test_crops, 
                                            steps=STEP_SIZE_TEST,
                                            verbose=1)

        np.save(open('predictions_siroviVGG16_test.npy', 'wb'), predictions)

    preds = np.argmax(predictions, axis=-1) # multiple categories

    label_map = (train_generator.class_indices)
    label_map = dict((v,k) for k,v in label_map.items()) # flip k,v
    preds_names = [label_map[k] for k in preds]

    print("Točnost: " + str((sum(preds == test_generator.classes)/len(preds))*100) + " %")

	top3_acc = top3_tocnost(predictions, test_generator)

    cm = confusion_matrix(test_generator.classes, preds))
    report = classification_report(test_generator.classes, preds))

    print("Scoreovi za pojedine autore:\n" + report)

    df_cm = pd.DataFrame(cm)
    plt.figure(figsize = (20,15))
    sns.set(font_scale=1.9)#for label size

    sns.set_style("darkgrid")

    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})# font size
    plt.show()
    plt.savefig("sirovi_vgg16_matrica_konfuzije.jpg")


model = initialize()
train(model, True)
test(model)

