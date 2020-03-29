import numpy as np
from sklearn.model_selection import KFold
import keras
from PIL import Image
import os
import matplotlib.pyplot as plt
from random import randint

# Global DATA 
pokemon =  ['bulbasaur', 'charmander', 'meowth', 'pikachu', 'squirtle']
data_x = 'data/x.npy'
data_y = 'data/y.npy'
history_file = 'data/hist.npy'
model_file = 'data/model.h5'

def getData():
    '''
    load data if exists or else creat dat from seed data
    '''
    if os.path.isfile(data_x) and os.path.isfile(data_y):
        x = np.load(data_x)
        y = np.load(data_y)
        return (x,y)
    else:
        x = []
        y = np.loadtxt("seed/keys.txt")
        # Hot one encoding
        y = keras.utils.to_categorical(y,5)
        numSeeds = y.shape[0]
        for i in range(1,numSeeds+1):
            img = np.array(Image.open("seed/pokemon"+str(i)+".bmp"))
            x.append(np.array(img))
        # Reshape and normailise x
        x = np.array(x).reshape(numSeeds,255,255,1)/255
        return (x,y)

def openMSPaint():
    '''
    Open MS paint with template image
    '''
    directory=r'img'
    name = "new_image.bmp"
    ImageEditorPath=r'C:\WINDOWS\system32\mspaint.exe'
    saveas=os.path.join(directory,name)
    editorstring='""%s" "%s"'% (ImageEditorPath,saveas) 
    os.system(editorstring)

def drawPokemon(y):
    '''
    display image of pokemon to draw
    open paint and return users drawing
    '''
    template =  Image.new( 'L', (255,255), "white")
    template.save("img/new_image.bmp")
    img = Image.open("StarterPokemon/"+pokemon[y]+".bmp")
    arrimg = np.array(img)
    print('Draw this Pokemon')
    plt.axis('off')
    plt.title(pokemon[y])
    plt.imshow(arrimg)
    plt.show()
    print('')
    openMSPaint()
    return Image.open("img/new_image.bmp")
    
def saveNewPokemon(new_x, new_y):
    '''
    add new pokemon to dataset 
    after playing game
    '''
    new_y = keras.utils.to_categorical(new_y,5)
    x,y = getData()
    x = np.append(x, new_x, axis=0)
    y = np.append(y, [new_y], axis=0)
    np.save(data_x, x)
    np.save(data_y, y)

def getModel():
    '''
    definition of our model
    '''
    model = keras.Sequential([
            keras.layers.Conv2D(32, (3,3), padding='same', activation='relu',
                                input_shape=(255,255,1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3,3), padding='same', activation='relu',
                                input_shape=(255,255,1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3,3), padding='same', activation='relu',
                                input_shape=(255,255,1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),        
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(5, activation='softmax'),
            ])
    model.compile(optimizer="adam", loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def createModel():
    '''
    Create new model from scractch
    '''
    x,y = getData()
    model = getModel()
    model.fit(x, y, epochs=8, batch_size=64, verbose=0)
    model.save(model_file)
    return model

def loadModel():
    '''
    open saved model if exists, otherwise create model
    '''
    if os.path.isfile(model_file):
        return keras.models.load_model(model_file)
    print ("model file not found creating model")
    return createModel()

def evalModel():
    '''
    Use kfold evaluation on model to determine
    accuracy and graph improvments
    '''
    print("Evaluting model with new data")
    x,y = getData()
    kfolds = KFold(n_splits=5, random_state=None, shuffle=False)
    num_games =  y.shape[0]
    kfold_acc = []
    for train_index, test_index in kfolds.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        m = getModel()
        m.fit(x_train, y_train, epochs=8, batch_size=64, verbose=0)
        kfold_acc.append(m.evaluate(x_test, y_test)[1])
        del m
    acc = np.mean(kfold_acc)
    print ('Current Accracy is: ',acc)
    hist = np.array([[num_games,acc]])
    if os.path.isfile(history_file):
        hist = np.load(history_file)
        hist = np.append(hist, [[num_games,acc]], axis=0)
    acc_list = hist[...,1]
    num_list = hist[...,0]
    plt.scatter(num_list,acc_list)
    plt.title("ConvNet Improvments")
    plt.ylabel("ConvNet Accuracy")
    plt.xlabel("Number of Pokemon Drawn")
    plt.show()
    np.save(history_file, hist)

def play():
    '''
    main game loop
    '''
    model = loadModel()
    
    #pick random pokemon
    new_pokemon = randint(0,4) 
    #get users drawing
    input_img = np.array(drawPokemon(new_pokemon)).reshape(1,255,255,1)/255
    # make prediction on users drawing
    pokemon_predict = model.predict(input_img)[0]
    pokemon_predict = np.argmax(pokemon_predict)
    print ("You drew this: ")
    plt.axis('off')
    plt.imshow(input_img[0].reshape(255,255), cmap='gray')
    plt.show()
    print ("I think that is a ", pokemon[pokemon_predict])
    if (pokemon_predict == new_pokemon):
        print ("I was correct!")
    else:
        print ("You are not very good at drawing pokemon")
    # add newly drawn poekmen to data set
    saveNewPokemon(input_img, new_pokemon)


def main():
    choice = True
    while choice:
        play()
        user_input = input("play again? (Y/N)")
        while (not(user_input.upper() == "Y" or user_input.upper() == "N")):
            user_input = input("Error, enter Y or N")
        choice = user_input.upper() == "Y"
    #'''
    #retrain convnet with new data from session
    createModel()
    evalModel()

if __name__ == "__main__":
    main()

