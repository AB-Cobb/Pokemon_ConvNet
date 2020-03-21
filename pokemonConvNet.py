import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score
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
    if os.path.isfile(data_x) and os.path.isfile(data_y):
        x = np.load(data_x)
        y = np.load(data_y)
        return (x,y)
    else:
        numSeeds = 11
        x = []
        y = np.loadtxt("seed/keys.txt")
        y = keras.utils.to_categorical(y,5)
        for i in range(1,numSeeds+1):
            img = np.array(Image.open("seed/pokemon"+str(i)+".bmp"))  
            x.append(np.array(img))
        x = np.array(x).reshape(numSeeds,255,255,1)
        return (x,y)

def openMSPaint():
    SaveDirectory=r'img'
    sName = "new_image"
    ImageEditorPath=r'C:\WINDOWS\system32\mspaint.exe'
    saveas=os.path.join(SaveDirectory,sName + '.bmp')
    editorstring='""%s" "%s"'% (ImageEditorPath,saveas) 
    os.system(editorstring)

def drawPokemon(y):
    template =  Image.new( 'L', (255,255), "white")
    template.save("img/new_image.bmp")
    img = Image.open("StarterPokemon/"+pokemon[y]+".bmp")
    arrimg = np.array(img)
    print('draw this pokemon')
    plt.axis('off')
    plt.title(pokemon[y])
    plt.imshow(arrimg)
    plt.show()
    print('')
    openMSPaint()
    return Image.open("img/new_image.bmp")
    
def saveNewPokemon(new_x, new_y):
    new_y = keras.utils.to_categorical(new_y,5)
    x,y = getData()
    x = np.append(x, new_x, axis=0)
    y = np.append(y, [new_y], axis=0)
    np.save(data_x, x)
    np.save(data_y, y)

def createModel():
    x,y = getData()
    model = keras.Sequential([
            keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(255,255,1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.2),
            keras.layers.Flatten(),        
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(5, activation='softmax'),
            ]) 
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(x,y, epochs=8, batch_size=64)
    return (model,hist)

def getModel():
    if os.path.isfile(model_file):
        return keras.models.load_model(model_file)
    model, h = createModel()
    return model

def play():
    model = getModel()
    new_pokemon = randint(0,4)
    input_img = np.array(drawPokemon(new_pokemon)).reshape(1,255,255,1)
    pokemon_predict = model.predict(input_img)[0]
    for i in range(0,5):
        if pokemon_predict[i] == 1:
            pokemon_predict = i
            break
    print ("You drew this: ")
    plt.axis('off')
    plt.imshow(input_img[0].reshape(255,255), cmap='gray')
    plt.show()
    print ("I think that is a ", pokemon[pokemon_predict])
    if (pokemon_predict == new_pokemon):
        print ("I was correct!")
    else:
        print ("You are not very good at drawing pokemon")
    saveNewPokemon(input_img, new_pokemon)

    

def retrain():
    m,h = createModel()
    m.save(model_file)
    num_games =  np.load(data_y).shape[0]
    acc = h.history['accuracy'][-1]
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

def main():
    choice = True
    while choice:
        play()
        user_input = input("play again? (Y/N)")
        while (not(user_input.upper() == "Y" or user_input.upper() == "N")):
            user_input = input("error")
        choice = user_input.upper() == "Y"
    #retrain convnet with new data from session
    retrain()

if __name__ == "__main__":
    main()

