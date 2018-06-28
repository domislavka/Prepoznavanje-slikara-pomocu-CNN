from __future__ import print_function
import warnings
from os import environ
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, Input, Flatten, Activation, Dropout, Dense
from keras.optimizers import Adam, SGD
from keras.initializers import glorot_normal
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.regularizers import l2
from keras import backend as K
from keras.models import Sequential, Model
from os import path, getcwd
import pandas as pd
import random
import numpy as np
import shutil
from time import time
from VisualizeFilters import *

filepathForWeights = 'transfer-vgg16-200-pretrained.h5'

def pretrain(train_generator, val_generator, num_artists):
    """
    Pretraining faza za mrežu Transfer_VGG16_200
    """

    vgg16 = VGG16(include_top=False, weights='imagenet')

    # zadnji sloj od vgg16 zamijenimo malom mrezom
    x = vgg16.output
    x = Dense(128, activation='sigmoid')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    preds_layer = Dense(num_artists, activation='softmax')(x)

    transfer_vgg16_200 = Model(inputs=vgg16.input, outputs=preds_layer)

    # zamrznemo slojeve pretrained mreze
    for layer in vgg16.layers:
        layer.trainable = False

    transfer_vgg16_200.summary()

    transfer_vgg16_200.compile(loss='categorical_crossentropy',
                        optimizer=SGD(lr=1e-3, momentum=0.9),
                        metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir='./GraphTransferVgg16-200', 
                         write_graph=True, 
                         write_images=True,
                         write_grads=False)

    checkpoint = ModelCheckpoint(filepath=filepathForWeights,
                                    monitor='val_acc',
                                    mode='max',
                                    save_best_only=True,
                                    save_weights_only=False,
                                    verbose=1,
                                    period=1)

    STEP_SIZE_TRAIN = train_generator.n/train_generator.batch_size
    STEP_SIZE_VALID = val_generator.n/val_generator.batch_size
    
    transfer_vgg16_200.fit_generator(train_generator, 
                                     steps_per_epoch=STEP_SIZE_TRAIN,
                                     epochs=20,
                                     validation_data=val_generator,
                                     validation_steps=STEP_SIZE_VALID,
                                     callbacks=[tensorboard, checkpoint])

    # spremimo pretrained model
    transfer_vgg16_200.save_weights('transfer_vgg16_test_200_tezine.h5')
    transfer_vgg16_200.save('transfer_vgg16_test_200.h5')


def finetune(train_generator, val_generator, num_artists):

    """
    Funkcija za fine-tuning mreze Transfer_VGG16_200
    """

    base_model = VGG16(include_top=False, weights='None')

    for layer in base_model.layers:
        layer.trainable = True
        
    xx = base_model.output
    xx = Dense(128)(xx)
    xx = GlobalAveragePooling2D()(xx)
    xx = Dropout(0.3)(xx)
    predictions = Dense(num_artists, activation='softmax')(xx)

    finetuned_vgg16_200 = Model(inputs=base_model.input, outputs=predictions)

    finetuned_vgg16_200.load_weights("transfer-vgg16-200-pretrained.h5")


    tensorboard = TensorBoard(log_dir='./FineTunedGraphTransferVgg16-200')
    
    filepath = 'vgg16-transfer-200_fine_tuned_model.h5'
    
    checkpoint = ModelCheckpoint(filepath,
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=False,
                                save_weights_only=False,
                                mode='auto',
                                period=1)

    finetuned_vgg16_200.compile(loss="categorical_crossentropy",
                                optimizer=SGD(lr=0.0001, momentum=0.9),
                                metrics=["accuracy"])

    STEP_SIZE_TRAIN = train_generator.n/train_generator.batch_size
    STEP_SIZE_VALID = val_generator.n/val_generator.batch_size

    finetuned_vgg16_200.fit_generator(train_generator,
                                    steps_per_epoch=STEP_SIZE_TRAIN,
                                    epochs=20,
                                    callbacks=[tensorboard, checkpoint],
                                    validation_data=val_generator,
                                    validation_steps=STEP_SIZE_VALID)

    # spremimo finetuned model
    finetuned_vgg16_200.save_weights('finetuned_transfer_vgg16_test_200_tezine.h5')
    finetuned_vgg16_200.save('finetuned_transfer_vgg16_test_200.h5')



def train_transferVGG16_200(train_generator, val_generator, num_artists):

    """
    Funkcija za treniranje mreze Transfer_VGG16_200
    """

    pretrain(train_generator, val_generator)
    finetune(train_generator, val_generator)



def test_transferVGG16_200(test_generator, num_artists):

    """
    Funkcija za testiranje mreze Transfer_VGG16_200
    Na kraju se ispiše točnost, filteri mreže i saliency mape se spreme u datoteku
    """

    base_model = VGG16(include_top=False, weights=None)
    x = base_model.output
    x = Dense(128)(x)
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_artists, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.load_weights("finetuned_transfer_vgg16_test_200_tezine.h5")

    STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

    predictions = model.predict_generator(test_generator, 
                                    steps=STEP_SIZE_TEST,
                                    workers=4,
                                    verbose=1)

    preds = np.argmax(predictions, axis=-1)
    print("Točnost: %f%" % (sum(preds == test_generator.classes)/len(preds))*100)


test_transferVGG16_200(test_generator, test_generator.classes)
