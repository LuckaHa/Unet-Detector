from model import *
from data import *

import cv2 as cv
import numpy as np
import os
import sys


# PREMENNE
test_dir = "data/test/image"
generated_dir = "data/test_generated"
if (len(sys.argv) > 1): # OD POUZIVATELA
    num_of_test_imgs = int(sys.argv[1])
else: # DEFAULT
    num_of_test_imgs = 9


# NACITAT MODEL
# struktura modelu
json_file = open('model/modelStructure.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# vahy modelu
model.load_weights("model/modelWeights.h5")
print("Model loaded from disk") 


# KOMPILACIA
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# SPUSTENIE NA TESTOVACICH VZORKACH
testGene = testGenerator(test_dir)
results = model.predict_generator(testGene,num_of_test_imgs,verbose=1)
saveResult(test_dir,results)
