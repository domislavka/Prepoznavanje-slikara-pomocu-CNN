from __future__ import print_function
import warnings
from os import environ
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras_preprocessing.image import save_img, load_img, img_to_array, array_to_img
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
from vis.visualization import visualize_activation, visualize_saliency, get_num_filters
from matplotlib import pyplot as plt
from vis.input_modifiers import Jitter
from top3_accuracy import *

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
STEP_SIZE_VALID = validation_generator.n/validation_generator.batch_size + 1
STEP_SIZE_TEST = test_generator.n//test_generator.batch_size + 1

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

    base_model = VGG16(include_top=False, weights=None)

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


def test_transferVGG16_300(loadSavedPreds = True):

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

    if loadSavedPreds == True:
        predictions = np.load('predictions_transf_vgg16_300_test.npy')

    else:
        predictions = model.predict_generator(test_crops,
                                            steps=STEP_SIZE_TEST,
                                            workers=4,
                                            verbose=1)

        np.save(open('predictions_transf_vgg16_300_test.npy', 'wb'), predictions)


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
    plt.savefig("transfer_vgg16_300_matrica_konfuzije.jpg")


    return model


model = test_transferVGG16_300()
model.summary()


#########################################
# maksimizacija nekog autora kroz mrežu #
#########################################

layer_idx = utils.find_layer_idx(model, 'predictions')
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

author_index = 20

plt.rcParams['figure.figsize'] = (18, 6)

img = visualize_activation(model, 
                            layer_idx, 
                            filter_indices=author_index, 
                            max_iter=500, 
                            input_modifiers=[Jitter(16)])
plt.imshow(img)
save_img('transfer_vgg16_300_max_' + str(author_index) + '.png', img)

################################
# maksimizacija za više autora #
################################

authors = np.random.permutation(1000)[:15]

vis_images = []
image_modifiers = [Jitter(16)]
for idx in categories:    
    img = visualize_activation(model, 
                                layer_idx, 
                                filter_indices=idx, 
                                max_iter=500, 
                                input_modifiers=image_modifiers)
    
    # Reverse lookup index to imagenet label and overlay it on the image.
    img = utils.draw_text(img, utils.get_imagenet_label(idx))
    vis_images.append(img)

# Generate stitched images with 5 cols (so it will have 3 rows).
plt.rcParams['figure.figsize'] = (50, 50)
stitched = utils.stitch_images(vis_images, cols=5)
plt.axis('off')
plt.imshow(stitched)
plt.show()
save_img('activation_max_transfer_300.png', stitches)

########################
# filteri conv slojeva #
########################

selected_indices = []
for layer_name in ['block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']:
    layer_idx = utils.find_layer_idx(model, layer_name)

    # Visualize all filters in this layer.
    filters = np.random.permutation(get_num_filters(model.layers[layer_idx]))[:10]
    selected_indices.append(filters)

    # Generate input image for each filter.
    vis_images = []
    for idx in filters:
        img = visualize_activation(model, layer_idx, filter_indices=idx)

        # Utility to overlay text on image.
        img = utils.draw_text(img, 'Filter {}'.format(idx))    
        vis_images.append(img)

    # Generate stitched image palette with 5 cols so we get 2 rows.
    stitched = utils.stitch_images(vis_images, cols=5)    
    plt.figure()
    plt.axis('off')
    plt.imshow(stitched)
    plt.show()

    save_img('transfer_vgg_300_' + layer_name + '.png', stitched)

################
# saliency map #
################

picasso = utils.load_img('/home/student1/ivsenki/Desktop/Vje-ba/testPicasso.jpg', target_size=(224, 224))

layer_idx = utils.find_layer_idx(model, 'predictions')
  
# 20 is the index corresponding to picasso
grad_picasso = visualize_saliency(transfer_vgg16, 
                                   layer_idx, 
                                   filter_indices=28,
                                   seed_input=picasso)

save_img('transfer_vgg16_300_saliency_picasso.png', grad_picasso)

#############################################
# testiraj mrežu na umjetno dorađenoj slici #
#############################################

imgFilePath = "golden_gate_matisse.png"
# imgFilePath = "golden_gate_starry.png"
# imgFilePath = "golden_gate_escher.png"

img = image.load_img(imgFilePath)
x = image.img_to_array(img)

img_cropped = center_crop(x, (224, 224))
img_cropped_array = image.array_to_img(img_cropped)
img_cropped_array.show()

x_pred = model.predict_classes(img_cropped.reshape(1, 224, 224, 3), batch_size=1)
print('Sliku je naslikao: ', label_map[x_pred[0]])

