
# Model mreže

# In[10]:


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



# koristimo adamov optimizator i metrika je točnost
#model.compile(loss='categorical_crossentropy',
#              optimizer=Adam(lr=1e-4),
#              metrics=['accuracy'])

# crta tablicu slojeva mreže
#model.summary()


# In[11]:


#model.load_weights('pokusaj300-20.h5')


# In[ ]:
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size

# treniramo mrežu....
#model.fit_generator(train_crops,
#                    steps_per_epoch=STEP_SIZE_TRAIN,
#                    epochs=10,
#                    validation_data=val_crops,
#                    validation_steps=STEP_SIZE_VALID,
#                    workers=4,
#                    callbacks=[tbCallBack])

#model.save_weights('prvi_pokusaj300SVE-SLIKE20.h5')


# In[89]:


#evaluation = model.evaluate_generator(test_crops,
#                         steps=test_generator.n//test_generator.batch_size,
#                         workers=4,
#                         verbose=1)

#print(model.metrics_names)
#print(evaluation)


# In[164]:


#predictions = model.predict_generator(test_crops, 
#                                      steps=test_generator.n//test_generator.batch_size,
#                                      workers=4,
#                                      verbose=1)

#preds = np.argmax(predictions, axis=-1) #multiple categories

#label_map = (train_generator.class_indices)
#label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
#preds_names = [label_map[k] for k in preds]


# In[168]:


#print(sum(preds == test_generator.classes)/len(preds))


# In[180]:


#print(label_map)


# In[184]:


# spremimo modelove "pogodtke"

#np.save(open('predictions_base_test.npy', 'wb'), predictions)


# # mali testovi kako je base model los

# In[182]:


#from keras.preprocessing import image

#img_path = 0
#img = image.load_img("/home/ivana/repos/Vje-ba/testPicasso.jpg")
#x = image.img_to_array(img)

#print(type(img))
#print(img.size)
#img_cropped = center_crop(x, (224, 224))
#img_cropped_blah = image.array_to_img(img_cropped)
#img_cropped_blah.show()

#x_pred = model.predict_classes(img_cropped.reshape(1, 224, 224, 3), batch_size=1)
#print('Sliku je naslikao: ', label_map[x_pred[0]])


# In[181]:


#img1 = image.load_img("/home/ivana/repos/Vje-ba/testGogh.jpg")
#x1 = image.img_to_array(img1)

#print(type(img1))
#img_cropped1 = center_crop(x1, (224, 224))
#img_cropped_blah1 = image.array_to_img(img_cropped1)
#img_cropped_blah1.show()

#x_pred1 = model.predict_classes(img_cropped1.reshape(1, 224, 224, 3), batch_size=1)
#print('Sliku je naslikao: ', label_map[x_pred1[0]])


# In[173]:


#from sklearn.metrics import classification_report, confusion_matrix

""" print(confusion_matrix(test_generator.classes, preds))
print(classification_report(test_generator.classes, preds)) """

def sirovi_VGG16():

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

    my_vgg16 = Model(inputs=base.input, outputs=preds_layer)

    return my_vgg16

def compile_siroviVGG16(model):

    model.compile(loss='categorical_crossentropy',
                         optimizer=Adam(lr=1e-4),
                         metrics=['accuracy'])

    model.summary()

def train_siroviVGG16(model):

    model.fit_generator(train_crops,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        epochs=17,
                        validation_data=val_crops,
                        validation_steps=STEP_SIZE_VALID,
                        workers=4,
                        callbacks=[tbCallBack, mdCheckPoint])

    # spremimo model
    model.save_weights('sirovi_vgg16_3.h5')
"""
model = sirovi_VGG16()
model.load_weights('sirovi-vgg16.weights2.01--0.10643.hdf5')
compile_siroviVGG16(model)
train_siroviVGG16(model)
"""
