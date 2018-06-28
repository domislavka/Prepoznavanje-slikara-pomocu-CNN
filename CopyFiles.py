import shutil
import pandas as pd
from os import path, getcwd, makedirs, walk


def createFolder(name):
    if not path.exists(name):
        makedirs(name)

moj_direktorij = getcwd()

train_dir = path.join(path.abspath(path.join(moj_direktorij, '..')), 'train200')
validation_dir = path.join(path.abspath(path.join(moj_direktorij, '..')), 'validation200')
test_dir = path.join(path.abspath(path.join(moj_direktorij, '..')), 'test200')

train2_dir = path.join(path.abspath(path.join(moj_direktorij, '..')), 'images')

# ako validation, test i train poddirektoriji ne postoje, stvori ih
createFolder(train_dir)
createFolder(validation_dir)
createFolder(test_dir)
    
fileovi = list(walk(train2_dir))[0][2]

def copyImagesToFiles(artist, artistImagesTrain, artistImagesVal, artistImagesTest):
        
    a_trdir = path.join(train_dir, artist)
    a_vadir = path.join(validation_dir, artist)
    a_tedir = path.join(test_dir, artist)
    
    createFolder(a_trdir)
    createFolder(a_vadir)
    createFolder(a_tedir)
    
    for item in artistImagesTrain.new_filename:
        if not path.exists(path.join(a_trdir, item)):
            source = path.join(train2_dir, item)
            dest = a_trdir
            shutil.copy(source, dest)
        
    for item in artistImagesVal.new_filename:
        if not path.exists(path.join(a_vadir, item)):
            source = path.join(train2_dir, item)
            dest = a_vadir
            shutil.copy(source, dest)
            
    for item in artistImagesTest.new_filename:
        if not path.exists(path.join(a_tedir, item)):
            source = path.join(train2_dir, item)
            dest = a_tedir
            shutil.copy(source, dest)
