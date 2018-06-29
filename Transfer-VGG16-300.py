from __future__ import print_function
import warnings
from os import environ
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Activation, Dropout, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import shutil
from VisualizeFilters import *
from dataLoad import *

filepathForWeights = 'transfer-vgg16-300-pretrained.h5'


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
STEP_SIZE_VALID = validation_generator.n/validation_generator.batch_size
STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

def pretrain():
    """
    Pretraining faza za mrežu Transfer_VGG16_300
    """

    vgg16 = VGG16(include_top=False, weights='imagenet')

    # zadnji sloj od vgg16 zamijenimo malom mrezom
    x = vgg16.output
    x = Dense(128, activation='sigmoid')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    preds_layer = Dense(num_artists, activation='softmax')(x)

    transfer_vgg16_300 = Model(inputs=vgg16.input, outputs=preds_layer)

    # zamrznemo slojeve pretrained mreze
    for layer in vgg16.layers:
        layer.trainable = False

    transfer_vgg16_300.summary()

    transfer_vgg16_300.compile(loss='categorical_crossentropy',
                               optimizer=SGD(lr=1e-3, momentum=0.9),
                               metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir='./GraphTransferVgg16-300', 
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
    
    transfer_vgg16_300.fit_generator(train_crops, 
                                     steps_per_epoch=STEP_SIZE_TRAIN,
                                     epochs=20,
                                     validation_data=val_crops,
                                     validation_steps=STEP_SIZE_VALID,
                                     callbacks=[tensorboard, checkpoint])

    # spremimo pretrained model
    transfer_vgg16_300.save_weights('transfer_vgg16_test_300_tezine.h5')
    transfer_vgg16_300.save('transfer_vgg16_test_300.h5')


def finetune():

    """
    Funkcija za fine-tuning mreze Transfer_VGG16_300
    """

    base_model = VGG16(include_top=False, weights='None')

    for layer in base_model.layers:
        layer.trainable = True
        
    xx = base_model.output
    xx = Dense(128)(xx)
    xx = GlobalAveragePooling2D()(xx)
    xx = Dropout(0.3)(xx)
    predictions = Dense(num_artists, activation='softmax')(xx)

    finetuned_vgg16_300 = Model(inputs=base_model.input, outputs=predictions)

    finetuned_vgg16_300.load_weights("transfer-vgg16-300-pretrained.h5")


    tensorboard = TensorBoard(log_dir='./FineTunedGraphTransferVgg16-300')
    
    filepath = 'vgg16-transfer-300_fine_tuned_model.h5'
    
    checkpoint = ModelCheckpoint(filepath,
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=False,
                                save_weights_only=False,
                                mode='auto',
                                period=1)

    finetuned_vgg16_300.compile(loss="categorical_crossentropy",
                                optimizer=SGD(lr=0.0001, momentum=0.9),
                                metrics=["accuracy"])

    finetuned_vgg16_300.fit_generator(train_crops,
                                    steps_per_epoch=STEP_SIZE_TRAIN,
                                    epochs=20,
                                    callbacks=[tensorboard, checkpoint],
                                    validation_data=val_crops,
                                    validation_steps=STEP_SIZE_VALID)

    # spremimo finetuned model
    finetuned_vgg16_300.save_weights('finetuned_transfer_vgg16_test_300_tezine.h5')
    finetuned_vgg16_300.save('finetuned_transfer_vgg16_test_300.h5')


def train_transferVGG16_300():

    """
    Funkcija za treniranje mreze Transfer_VGG16_300
    """

    pretrain()
    finetune()


def test_transferVGG16_300():

    """
    Funkcija za testiranje mreze Transfer_VGG16_300
    Na kraju se ispiše točnost
    Vraća model mreze Transfer_VGG16_300
    """

    base_model = VGG16(include_top=False, weights=None)
    x = base_model.output
    x = Dense(128)(x)
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_artists, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.load_weights("finetuned_transfer_vgg16_test_300_tezine.h5")

    predictions = model.predict_generator(test_crops,
                                        steps=STEP_SIZE_TEST,
                                        workers=4,
                                        verbose=1)

    np.save(open('predictions_transf_vgg16_300_test.npy', 'wb'), predictions)

    preds = np.argmax(predictions, axis=-1)
    print("Točnost: %f %" % (sum(preds == test_generator.classes)/len(preds))*100)

    # treba malo proljepšati izgled matrica konfuzije

    print(confusion_matrix(test_generator.classes, preds))
    print(classification_report(test_generator.classes, preds))

    return model


model = test_transferVGG16_300()
model.summary()

# filteri layer-a spremamo u datoteku
layer_name = 'neko_ime_layera'
layer_index = neki_broj_layera

img_width = 224
img_height = 224

num_filters = model.layers[layer_index].output.shape[3]

layerFilters(model, layer_name, img_width, img_height, num_filters)

# maksimizacija nekog autora kroz mrežu

# SALIENCY MAP

# PODMETNI UMJETNO DORAĐENU SLIKU

'''
OVO NAM SLUZI DA ZNAMO KAKO CEMO NA NAJBOLJOJ MREZI
UCITAT UMJETNO DORADJENU SLIKU I NAPRAVIT PREDICT


img = image.load_img("testPicasso.jpg")
x = image.img_to_array(img)

img_cropped = center_crop(x, (224, 224))
img_cropped_blah = image.array_to_img(img_cropped)
img_cropped_blah.show()

x_pred = model.predict_classes(img_cropped.reshape(1, 224, 224, 3), batch_size=1)
print('Sliku je naslikao: ', label_map[x_pred[0]])

'''

