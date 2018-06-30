from __future__ import print_function
import warnings
from os import environ
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.initializers import glorot_normal
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.regularizers import l2
from keras import backend as K
from keras.models import Sequential
from keras.preprocessing import image
from os import path, getcwd
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import random
import numpy as np
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
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

# moramo staviti +1 jer 1710 nije djeljivo s 60
STEP_SIZE_VALID = validation_generator.n/validation_generator.batch_size + 1

# moramo staviti +1 jer 1710 nije djeljivo s 60
STEP_SIZE_TEST = test_generator.n//test_generator.batch_size + 1

# velicina slika koje dajemo ulaznom sloju mreze
input_shape = (224, 224, 3)

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
        model.load_weights('Baseline-300.h5')
        return model

    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size

    # treniramo mrežu....
    model.fit_generator(train_crops,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        epochs=20,
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

    model.save_weights('Baseline-weights.h5')
    return model

def test(model, loadSavedPreds = True):

    '''
    Funkcija za testiranje mreze Baseline
    '''

    if loadSavedPreds == True:
        predictions = np.load('predictions_base_test.npy')
    
    else:

        predictions = model.predict_generator(test_crops, 
                                            steps=STEP_SIZE_TEST,
                                            workers=4,
                                            verbose=1)

        np.save(open('predictions_base_test.npy', 'wb'), predictions)

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

    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16})# font size
    plt.show()
    plt.savefig("baseline_matrica_konfuzije.jpg")


model = initialize()
model.summary()
train(model, True)
test(model)


