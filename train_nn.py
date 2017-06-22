#!/usr/bin/env python

from glob import glob
import numpy as np
import pandas as pd
import sklearn
from random import shuffle
#import seaborn as sns
import cv2
import matplotlib.pyplot as plt
from keras.layers import Flatten,Dense,Convolution2D,MaxPooling2D,Dropout,Activation,Lambda,Cropping2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint
experiment=3
input_shape=(64,48,3)#(128,96,3)#640,480,3
batch_size=64
def lenet(input_shape=input_shape):
    """LeNet architecture(final solution)"""
    model=Sequential()
    model.add(Lambda(lambda x: (x-128.0)/128.0,input_shape=input_shape))
    #model.add(Cropping2D(cropping=((70,25), (0,0))))

    # SOLUTION: Layer 1: Convolutional. Input =  Output = 
    model.add(Convolution2D(nb_filter=32, nb_row=5,nb_col=5, subsample=(1,1),bias=True))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode="valid"))

    # SOLUTION: Layer 2: Convolutional. Output = 
    model.add(Convolution2D(nb_filter=32, nb_row=5,nb_col=5, subsample=(1,1),bias=True))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode="valid"))

    model.add(Convolution2D(nb_filter=64, nb_row=5,nb_col=5, subsample=(1,1),bias=True))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode="valid"))

    
 #   model.add(Dropout(0.25))

    # SOLUTION: Flatten. Input = 5x5x16. Output = .
    model.add(Flatten())
    
    # SOLUTION: Layer 3: Fully Connected. Input = . Output = .
    model.add(Dense(120,bias=True))
    
    # SOLUTION: Activation.
    model.add(Activation('relu'))

    model.add(Dropout(0.25))
    
    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    model.add(Dense(84,bias=True))
    
    # SOLUTION: Activation.
    model.add(Activation('relu'))

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 3.
    model.add(Dense(3,activation="softmax"))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

from common_functions import get_label,timeit
label_binarizer = LabelBinarizer()
label_binarizer.fit([0,1,2])

def load_image(filepath):
    img=cv2.imread(filepath)
    if img is None:
        return None
    
    img=img[...,::-1]#flip BGR to RGB
    # Crop the image
    top=int(img.shape[0]*0.15)
    bottom=int(img.shape[0]*0.85)
    img=img[top:bottom,:,:]
    
    rows=input_shape[0]
    cols=input_shape[1]
    cv2_shape=(cols,rows)
    color1 = cv2.resize(img[:,:,0],cv2_shape )
    color2 = cv2.resize(img[:,:,1], cv2_shape)
    color3 = cv2.resize(img[:,:,2], cv2_shape)
    img=np.dstack((color1, color2, color3))
    return img
    
def generator(samples, batch_size=batch_size):
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            labels = []
            for filepath in batch_samples:
                #Data Augmentation 2: Flipping Image Laterally
                #flipped_image = np.fliplr(original_image)
                img=load_image(filepath)
                if img is None:
                    continue
                images.append(img)
                labels.append(get_label(filepath))

# trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(labels)
            
            y_train = label_binarizer.transform(y_train)
            yield sklearn.utils.shuffle(X_train, y_train)


def test_generator(samples, batch_size=batch_size):
    num_samples = len(samples)

    for offset in range(0, num_samples, batch_size):
        start=offset
        end=min(offset+batch_size,num_samples)
        print("start","end",start,end)
        batch_samples = samples[start:end]
        images = []
        for filepath in batch_samples:
                #Data Augmentation 2: Flipping Image Laterally
                #flipped_image = np.fliplr(original_image)
            img=load_image(filepath)
            if img is None:
                continue
            images.append(img)

        X_test = np.array(images)
        yield X_test

def train_model(model,train_samples,validation_samples,batch_size):
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
    checkpoint = ModelCheckpoint("best_nn_weights.h5",
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='auto')
    callbacks_list=[checkpoint]
    history=model.fit_generator(train_generator, 
                                samples_per_epoch= len(train_samples), 
                                validation_data=validation_generator,
                                nb_val_samples=len(validation_samples), nb_epoch=50,
                                callbacks=callbacks_list)
    return history

@timeit
def main(base,samples=5):
    test_files = glob(base+"test/*.jpg")
    test_files+= glob(base+"test_stg2/*.jpg")

    train_files_one = glob(base+"train/Type_1/*.jpg")+glob(base+"additional/Type_1/*.jpg")
    train_files_two = glob(base+"train/Type_2/*.jpg")+glob(base+"additional/Type_2/*.jpg")
    train_files_three = glob(base+"train/Type_3/*.jpg")+glob(base+"additional/Type_3/*.jpg")
    # Additional files

    shuffle(train_files_one)
    shuffle(train_files_two)
    shuffle(train_files_three)
    train_files=train_files_one[0:samples]+train_files_two[0:samples]+train_files_three[0:samples]
    train_samples, validation_samples = train_test_split(train_files, test_size=0.2)
    old_model_name=None
    model_name="model_v%s"%experiment
    model=lenet()
    if old_model_name:
        model.load_weights(old_model_name+"_weights.h5")

    print("train_samples",len(train_samples),train_samples[0])
    print("validation_samples",len(validation_samples),validation_samples[0])

    history=train_model(model,train_samples,validation_samples,batch_size=batch_size)
    model.save(model_name+".h5")
    model.save_weights(model_name+"_weights.h5")

    tg = test_generator(test_files, batch_size=batch_size)
    predictions_df=pd.DataFrame()
    for X_test in tg:
        output=model.predict(X_test)
        test_image_df=pd.DataFrame({"Type_1":output[:,0],
                                    "Type_2":output[:,1],
                                    "Type_3":output[:,2],
                                    })
        predictions_df=predictions_df.append(test_image_df,ignore_index=True)

    predictions_df["imagepath"]=test_files
    func=lambda x: x.split("/")[-1]
    predictions_df["image_name"]=predictions_df["imagepath"].apply(func)
    predictions_df[["image_name","Type_1","Type_2","Type_3"]].to_csv("nn_v%s.csv"%experiment,
                                                                     index=False)

if __name__=="__main__":
    basepath="/data/"
    main(basepath,1000)
