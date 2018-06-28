
# coding: utf-8

# In[1]:


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


# In[2]:


def sve_jpg(df):
    for item in list(df.new_filename):
        if not '.jpg' in item:
            return False
    return True


# In[3]:


df = pd.read_csv(path.join(getcwd(), 'all_data_info.csv'))
seed = 123

print(df.shape)
df.head()


# In[4]:


print('Sve su jpg: ' + str(sve_jpg(df)))


# In[5]:


threshold = 300

x = list(df['artist'].value_counts())
# broj umjetnika koji imaju vise ili jednako od 300 slika
print(len([a for a in x if a >= threshold]))
# len(set(x)) #---> ukupan broj umjetnika


# In[6]:


# train, validation, test --- 80, 10, 10
num_train = 240
num_val = 30
num_test = num_val
num_samples = num_train + num_val + num_test
b_size = 60

#lista umjetnika koje ćemo promatrati
temp = df['artist'].value_counts()
artists = temp[temp >= threshold].index.tolist()
# print(artists)

num_artists = len(artists)
print('Prepoznajemo ' + str(num_artists) + ' umjetnika')


# In[7]:


#train_dfs = []
#val_dfs = []
#test_dfs = []

#for a in artists:
    # PROVJERI KASNIJE ŠTA JE S NA=TRUE
#    tmp = df[df['artist'].str.startswith(a)].sample(n=num_samples, random_state=seed)
    # print(tmp.shape)
#    t_df = tmp.sample(n=num_train, random_state=seed)
#    rest_df = tmp.loc[~tmp.index.isin(t_df.index)] # uzmi komplement od t_df
    # print(rest_df.shape)
#    v_df = rest_df.sample(n=num_val, random_state=seed)
#    te_df = rest_df.loc[~rest_df.index.isin(v_df.index)]
    
#    train_dfs.append(t_df)
#    val_dfs.append(v_df)
#    test_dfs.append(te_df)
    
    # ovo se pokrene samo jednom!!
#    copyImagesToFiles(a, t_df, v_df, te_df)

#train_df = pd.concat(train_dfs)
#val_df = pd.concat(val_dfs)
#test_df = pd.concat(test_dfs)

#print('train tablica\t\t', train_df.shape)
#print('validation tablica\t', val_df.shape)
#print('test tablica\t\t', test_df.shape)


# In[162]:


def center_crop(img, center_crop_size):
    assert img.shape[2] == 3
    centerw, centerh = img.shape[0] // 2, img.shape[1] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    return img[centerw-halfw:centerw+halfw, centerh-halfh:centerh+halfh, :]

# https://jkjung-avt.github.io/keras-image-cropping/
def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]


def crop_generator(batches, crop_length, random_cropping=True, test_batch=False):
    '''
    Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator
    '''
    while True:
        if test_batch == False:
            batch_x, batch_y = next(batches)
        else:
            batch_x = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        for i in range(batch_x.shape[0]):
            if random_cropping == True:
                batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
            else:
                batch_crops[i] = center_crop(batch_x[i], (crop_length, crop_length))
        if test_batch == False:
            yield (batch_crops, batch_y)
        else:
            yield batch_crops


# In[163]:


# velicina slika koje dajemo ulaznom sloju mreze
input_shape = (224, 224, 3)
# velicina batch-a
b_size = 30

train_datagen = ImageDataGenerator(
                horizontal_flip=True)

val_datagen = ImageDataGenerator(
                horizontal_flip=True)
test_datagen = ImageDataGenerator()


train_generator = train_datagen.flow_from_directory(
                    '../train',
                    batch_size=b_size,
                    class_mode='categorical')
train_generator = train_datagen.standardize(train_generator)
# na slikama iz train skupa radimo crop na slučajnom mjestu
train_crops = crop_generator(train_generator, 224)

validation_generator = val_datagen.flow_from_directory(
                    '../validation',
                    batch_size=b_size,
                    class_mode='categorical')
# na slikama iz validation skupa radimo centralni crop
val_crops = crop_generator(validation_generator, 224, False)


test_generator = test_datagen.flow_from_directory(
                '../test',
                batch_size=b_size,
                class_mode=None, # this means our generator will only yield batches of data, no labels
                shuffle=False) # our data will be in order

test_crops = crop_generator(test_generator, 224, False, True)


# Model mreže

# In[10]:

<<<<<<< HEAD
###
###
### Baseline mreža, insprirana glavnim člankom
###
###

model = Sequential(name='baseline')

model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', input_shape=input_shape, kernel_initializer=glorot_normal()))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(32, kernel_size=3, strides=2, padding="same", input_shape=input_shape, kernel_initializer=glorot_normal()))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(4*num_artists, input_shape=(6272,), kernel_initializer=glorot_normal()))
model.add(Activation('relu'))
model.add(Dense(num_artists, input_shape=(4*num_artists,), activation='softmax', kernel_initializer=glorot_normal()))
=======

# model mreže inspiriran glavnim člankom

#model = Sequential()

#model.add(Conv2D(32, 
#                 kernel_size=3, 
#                 strides=2, 
#                 padding='same', 
#                 input_shape=input_shape,
#                 kernel_initializer=glorot_normal()))
#model.add(Activation('relu'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=2))
#model.add(Conv2D(32, 
#                 kernel_size=3, 
#                 strides=2, 
#                 padding="same", 
#                 input_shape=input_shape,
#                 kernel_initializer=glorot_normal()))
#model.add(Activation('relu'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=2))
#model.add(Flatten())
#model.add(Dense(4*num_artists, 
#                input_shape=(6272,),
#                kernel_initializer=glorot_normal()))
#model.add(Activation('relu'))
#model.add(Dense(num_artists, 
#                input_shape=(4*num_artists,),
#                activation='softmax',
#                kernel_initializer=glorot_normal()))

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
>>>>>>> b98b5e49a4cfad71860f2bb176e5ceaed833c03d

# koristimo adamov optimizator i metrika je točnost
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-4),
              metrics=['accuracy'])

# crta tablicu slojeva mreže
model.summary()


# In[11]:

# mrežu smo već istrenirali i spremili njene težine
model.load_weights('pokusaj300-20.h5')


# In[ ]:
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


# In[89]:


evaluation = model.evaluate_generator(test_crops,
                         steps=test_generator.n//test_generator.batch_size,
                         workers=4,
                         verbose=1)

print(model.metrics_names)
print(evaluation)


# In[164]:


predictions = model.predict_generator(test_crops, 
                                      steps=test_generator.n//test_generator.batch_size,
                                      workers=4,
                                      verbose=1)

preds = np.argmax(predictions, axis=-1) #multiple categories

label_map = (train_generator.class_indices)
label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
preds_names = [label_map[k] for k in preds]


# In[168]:


print(sum(preds == test_generator.classes)/len(preds))


# In[180]:


print(label_map)


# In[184]:


# spremimo modelove "pogotke"

np.save(open('predictions_base_test.npy', 'wb'), predictions)


# # mali testovi kako je base model los

# In[182]:


from keras.preprocessing import image

img_path = 0
img = image.load_img("/home/ivana/repos/Vje-ba/testPicasso.jpg")
x = image.img_to_array(img)

print(type(img))
print(img.size)
img_cropped = center_crop(x, (224, 224))
img_cropped_blah = image.array_to_img(img_cropped)
img_cropped_blah.show()

x_pred = model.predict_classes(img_cropped.reshape(1, 224, 224, 3), batch_size=1)
print('Sliku je naslikao: ', label_map[x_pred[0]])


# In[181]:


img1 = image.load_img("/home/ivana/repos/Vje-ba/testGogh.jpg")
x1 = image.img_to_array(img1)

print(type(img1))
img_cropped1 = center_crop(x1, (224, 224))
img_cropped_blah1 = image.array_to_img(img_cropped1)
img_cropped_blah1.show()

x_pred1 = model.predict_classes(img_cropped1.reshape(1, 224, 224, 3), batch_size=1)
print('Sliku je naslikao: ', label_map[x_pred1[0]])


# In[173]:


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(test_generator.classes, preds))
print(classification_report(test_generator.classes, preds))


###
###
### VGG-16 trenirana ispočetka
###
###

img_input = Input((224, 224, 3))

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

sirovi_vgg16 = Model(inputs=base.input, outputs=preds_layer, name='scratchVGG16')

<<<<<<< HEAD
sirovi_vgg16.compile(loss='categorical_crossentropy',
                     optimizer=SGD(lr=1e-3, momentum=0.9),
=======
my_vgg16.compile(loss='categorical_crossentropy',
                     optimizer=Adam(lr=1e-4),
>>>>>>> b98b5e49a4cfad71860f2bb176e5ceaed833c03d
                     metrics=['accuracy'])

sirovi_vgg16.summary()

sirovi_vgg16.fit_generator(train_crops,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=3,
                    validation_data=val_crops,
                    validation_steps=STEP_SIZE_VALID,
<<<<<<< HEAD
                    callbacks=[
                        TensorBoard(log_dir='./GraphSiroviVGG16', 
                                    histogram_freq=1, 
                                    write_graph=True, 
                                    write_images=True,
                                    write_grads=False),
                        ModelCheckpoint(filepath='sirovivgg16.weights.{epoch:02d}--{val_acc:.5f}.hdf5',
                                    monitor='val_acc',
                                    mode='max',
                                    save_best_only=False,
                                    save_weights_only=False,
                                    verbose=1,
                                    period=1)])
=======
                    workers=4,
                    callbacks=[tbCallBack, mdCheckPoint])
>>>>>>> b98b5e49a4cfad71860f2bb176e5ceaed833c03d

# spremimo model
my_vgg16.save_weights('sirovi_vgg16_1-3.h5')


###
###
### VGG-16 s prijenosom znanja
###
###

from keras.applications import vgg16

vgg16 = vgg16.VGG16(include_top=False, weights='imagenet')

x = vgg16.output
x = Dense(128, activation='sigmoid')(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
preds_layer = Dense(num_artists, activation='softmax')(x)

transfer_vgg16 = Model(inputs=vgg16.input, outputs=preds_layer, name='transferVGG16')

for layer in vgg16.layers:
    layer.trainable = False

transfer_vgg16.summary()

transfer_vgg16.compile(loss='categorical_crossentropy',
                     optimizer=SGD(lr=1e-3, momentum=0.9),
                     metrics=['accuracy'])

transfer_vgg16.fit_generator(train_crops,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=3,
                    validation_data=val_crops,
                    validation_steps=STEP_SIZE_VALID,
                    workers=4,
                    callbacks=[
                        TensorBoard(log_dir='./GraphTransfVGG16', 
                                    histogram_freq=1, 
                                    write_graph=True, 
                                    write_images=True,
                                    write_grads=False),
                        ModelCheckpoint(filepath='transfvgg16.weights.{epoch:02d}--{val_acc:.5f}.hdf5',
                                    monitor='val_acc',
                                    mode='max',
                                    save_best_only=False,
                                    save_weights_only=False,
                                    verbose=1,
                                    period=1)])

# spremimo model

<<<<<<< HEAD
transfer_vgg16.save_weights('transfer_vgg16_test.h5')
=======
transfer_vgg16.save_weights('transfer_vgg16_test.h5') """
>>>>>>> b98b5e49a4cfad71860f2bb176e5ceaed833c03d
