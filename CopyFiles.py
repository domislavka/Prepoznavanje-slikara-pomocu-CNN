import shutil
import pandas as pd
from os import path, getcwd, makedirs, walk

def createFolder(name):
    if not path.exists(name):
        makedirs(name)
        
def copyImage(check, source, dest):
    if not path.exists(check):
        copy(source, dest)
        
moj_direktorij = getcwd()

train_dir = path.join(path.abspath(path.join(moj_direktorij, '..')), 'train')
validation_dir = path.join(path.abspath(path.join(moj_direktorij, '..')), 'validation')
test_dir = path.join(path.abspath(path.join(moj_direktorij, '..')), 'test')

train2_dir = path.join(path.abspath(path.join(moj_direktorij, '..')), 'train_2')

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
        copyImage(path.join(a_trdir, item), path.join(train2_dir, item), a_trdir)
        
    for item in artistImagesVal.new_filename:
        copyImage(path.join(a_vadir, item), path.join(train2_dir, item), a_vadir)

    for item in artistImagesTest.new_filename:
        copyImage(path.join(a_tedir, item), path.join(train2_dir, item), a_tedir)