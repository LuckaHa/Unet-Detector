from model import *
from data import *
import matplotlib.pyplot as plt
import json
import sys

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# PREMENNE
from_user = sys.argv
if (len(from_user) > 1): # OD POUZIVATELA
    label_dir = from_user[1] # label, label_area
    steps = int(from_user[2])
    epochs = int(from_user[3])
    mode = from_user[4] # create, load
    test = from_user[5] # yes, no
    num_of_test_imgs = int(from_user[6])
else: # DEFAULT
    label_dir = 'label'
    steps = 25
    epochs = 40
    mode = 'create'
    test = 'yes'
    num_of_test_imgs = 7
train_dir = 'data/train'
test_dir = 'data/test/image'

print('Sample folder: ' + label_dir)
print('Batch size: ' + str(steps))
print('Number of epochs: ' + str(epochs))
print('Mode: ' + mode)
print('Run testing: ' + test)


# GENEROVANIE DAT
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='reflect')
myGene = trainGenerator(2,train_dir,'image',label_dir,data_gen_args,save_to_dir = None)


# VYTVORENIE U-NET a) nanovo
def create_unet(myGene):
    model = unet()
    model_checkpoint = ModelCheckpoint('model/unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
    history = model.fit_generator(myGene,steps_per_epoch=steps,epochs=epochs,callbacks=[model_checkpoint])
    # ulozit historiu
    with open('model/history.json', 'w') as f:
        json.dump(history.history, f)
    return model, history.history

# b) zo suboru
def load_unet():
    model = unet('model/unet_membrane.hdf5')
    # nacitat historiu
    history_dict = json.load(open('model/history.json', 'r'))
    return model, history_dict

if (mode == 'create'):
    model, history = create_unet(myGene) # a)
else:
    model, history = load_unet() # b)


# ULOZENIE MODELU
def save_model(model):
    model_json = model.to_json() # struktura
    with open("model/modelStructure.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model/modelWeights.h5") # vahy
    model.save("model/ourModel.h5")
    print("Model saved to disk")
save_model(model)


# VIZUALIZACIA VYVOJU PRESNOSTI A CHYBY
def visualize_accuracy(history_dict):
    plt.plot(history_dict['acc'])
    plt.title('Presnosť modelu')
    plt.ylabel('presnosť')
    plt.xlabel('epocha')
    plt.legend(['trénovanie'], loc='upper left')
    plt.savefig('model/accuracy.png')

    plt.plot(history_dict['loss'])
    plt.title('Chyba modelu')
    plt.ylabel('chyba')
    plt.xlabel('epocha')
    plt.legend(['trénovanie'], loc='upper left')
    plt.savefig('model/loss.png')
visualize_accuracy(history)


# TESTOVANIE (da sa spustit aj zvlast pomocou use_model.py)
def testing(test_dir):
    testGene = testGenerator(test_dir)
    results = model.predict_generator(testGene,num_of_test_imgs,verbose=1)
    saveResult(test_dir,results)
if (test == 'yes'):
    testing(test_dir)
