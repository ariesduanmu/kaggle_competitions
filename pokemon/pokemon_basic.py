from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

from PIL import Image

import os

import matplotlib.pyplot as plt

TRAIN_PATH = "pokemon_train.npy"
TEST_PATH = "pokemon_test.npy"

TRAIN_DIR = "data/train"
VALIDATION_DIR = "data/validation"

SEED = 113

def draw_history(history, filename_padding):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    
    plt.clf()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(f'training_and_validation_accuracy_{filename_padding}.png')
    
    plt.clf()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(f'training_and_validation_loss_{filename_padding}.png')

# some stupid solution store to image
def load_data():
    # load data
    train_data = np.load(TRAIN_PATH)
    # test_data = np.load(TEST_PATH)

    # random
    rng = np.random.RandomState(SEED)
    indices = np.arange(len(train_data))
    rng.shuffle(indices)
    train_data = train_data[indices]

    for i in range(len(train_data)):
        x = train_data[i][1:]
        y = train_data[i][0]

        d = f"data/train/{y}"
        if i < 100:
            d = f"data/validation/{y}"

        if not os.path.exists(d):
            os.mkdir(d)
        img = Image.fromarray(np.reshape(x, (128,128,3)), 'RGB')
        img.save(f'{d}/{i}.jpg')

def preprocess_data():
    train_data = np.load(TRAIN_PATH)
    # test_data = np.load(TEST_PATH)

    # random
    rng = np.random.RandomState(SEED)
    indices = np.arange(len(train_data))
    rng.shuffle(indices)
    train_data = train_data[indices]

    val_data = train_data[:100]

    train_data = train_data[100:]

    x_train = train_data[:,1:]
    y_train = train_data[:,0]

    x_val = val_data[:,1:]
    y_val = val_data[:,0]

    return x_train, y_train, x_val, y_val

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    print(model.summary())
    return model

def main():
    model = build_model()
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                       target_size=(128,128),
                                       batch_size=20,
                                       class_mode='categorical')
    validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                        target_size=(128,128),
                                        batch_size=20,
                                        class_mode='categorical')

    history = model.fit_generator(train_generator,
                         steps_per_epoch=100,
                         epochs=30,
                         validation_data=validation_generator,
                         validation_steps=20)
    model.save('pokemon.h5')
    draw_history(history, "basic")
    

def train_data():
    model = build_model()
    x_train, y_train, x_val, y_val = preprocess_data()
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)


    x_train = np.reshape(x_train, (x_train.shape[0], 128,128,3))
    x_val = np.reshape(x_val, (x_val.shape[0], 128,128,3))
    
    train_datagen.fit(x_train)
    validation_datagen.fit(x_val)

    history = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=20),
                                  steps_per_epoch=len(x_train) / 20, 
                                  epochs=30,
                                  validation_data=validation_datagen.flow(x_val, y_val, batch_size=20),
                                  validation_steps=20)
    model.save('pokemon_data.h5')
    draw_history(history, "basic_data")

if __name__ == "__main__":
    # main()
    train_data()






