from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from os import path, getcwd
from CopyFiles import *
from imageAugmentation import *

df = pd.read_csv(path.join(getcwd(), 'all_data_info.csv'))
seed = 123
b_size = 60

def prepare(threshold=300, copyFiles=False):

    """
    Funkcija vraca ukupan broj autora koje ćemo prepoznavati
    ako je copyFiles True, onda ce dataset podijeliti u train, 
    validation i test foldere
    """

    # train, validation, test --- 80, 10, 10
    num_train = threshold*0.8
    num_val = (threshold - num_train) // 2
    num_test = num_val
    num_samples = num_train + num_val + num_test

    #lista umjetnika koje ćemo promatrati
    temp = df['artist'].value_counts()
    artists = temp[temp >= threshold].index.tolist()

    num_artists = len(artists)
    print('Prepoznajemo ' + str(num_artists) + ' umjetnika')

    if copyFiles == True:
        for a in artists:
            tmp = df[df['artist'].str.startswith(a)].sample(n=num_samples, random_state=seed)
            t_df = tmp.sample(n=num_train, random_state=seed)
            rest_df = tmp.loc[~tmp.index.isin(t_df.index)] # uzmi komplement od t_df
            v_df = rest_df.sample(n=num_val, random_state=seed)
            te_df = rest_df.loc[~rest_df.index.isin(v_df.index)]
            
            # ovo se pokrene samo jednom!!
            copyImagesToFiles(a, t_df, v_df, te_df)
    
    return num_artists

def loadTrain():

    """
    Funkcija vraća imagedatagenerator za train skup
    """

    train_datagen = ImageDataGenerator(horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
                        '../train',
                        batch_size=b_size,
                        class_mode='categorical')
    train_generator = train_datagen.standardize(train_generator)
    # na slikama iz train skupa radimo crop na slučajnom mjestu
    train_crops = crop_generator(train_generator, 224)

    return [train_generator, train_crops]


def loadVal():

    """
    Funkcija vraća imagedatagenerator za validation skup
    """

    val_datagen = ImageDataGenerator(horizontal_flip=True)
    validation_generator = val_datagen.flow_from_directory(
                        '../validation',
                        batch_size=b_size,
                        class_mode='categorical')
    # na slikama iz validation skupa radimo centralni crop
    val_crops = crop_generator(validation_generator, 224, False)

    return [validation_generator, val_crops]


def loadTest():

    """
    Funkcija vraća imagedatagenerator za test skup
    """

    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(
                    '../test',
                    batch_size=b_size,
                    class_mode=None, # this means our generator will only yield batches of data, no labels
                    shuffle=False)   # our data will be in order
    # na slikama iz test skupa radimo centralni crop
    test_crops = crop_generator(test_generator, 224, False, True)

    return [test_generator, test_crops]