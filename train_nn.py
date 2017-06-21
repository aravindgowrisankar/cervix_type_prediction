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
# TODO: Build the Final Test Neural Network in Keras Here
def lenet(input_shape=(640,480,3)):
    """LeNet architecture(final solution)"""
    model=Sequential()
    model.add(Lambda(lambda x: (x-128.0)/128.0,input_shape=input_shape))
    #model.add(Cropping2D(cropping=((70,25), (0,0))))

    # SOLUTION: Layer 1: Convolutional. Input =  Output = 
    model.add(Convolution2D(nb_filter=6, nb_row=5,nb_col=5, subsample=(1,1),bias=True))

    # SOLUTION: Activation.
    model.add(Activation('relu'))


    # SOLUTION: Pooling. Input =  Output = 
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode="valid"))

    # SOLUTION: Layer 2: Convolutional. Output = 
    model.add(Convolution2D(nb_filter=16, nb_row=5,nb_col=5, subsample=(1,1),bias=True))
    
    # SOLUTION: Activation.
    model.add(Activation('relu'))

    # SOLUTION: Pooling. Input =  Output = 
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode="valid"))

    # SOLUTION: Flatten. Input = 5x5x16. Output = .
    model.add(Flatten())
    
    # SOLUTION: Layer 3: Fully Connected. Input = . Output = .
    model.add(Dense(120,bias=True))
    
    # SOLUTION: Activation.
    model.add(Activation('relu'))

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
def generator(samples, batch_size=32):
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            labels = []
            for filepath in batch_samples:
                #Data Augmentation 2: Flipping Image Laterally
                img=cv2.imread(filepath)
                if img is None:
                    continue

                img=img[...,::-1]#flip BGR to RGB
                rows=640
                cols=480
                cv2_shape=(cols,rows)
                color1 = cv2.resize(img[:,:,0],cv2_shape )
                color2 = cv2.resize(img[:,:,1], cv2_shape)
                color3 = cv2.resize(img[:,:,2], cv2_shape)
                img=np.dstack((color1, color2, color3))
                
                #flipped_image = np.fliplr(original_image)
                images.append(img)
                labels.append(get_label(filepath))

# trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(labels)
            
            y_train = label_binarizer.transform(y_train)
            print("X_train.shape",X_train.shape)
            print("y_train.shape",y_train.shape)
            yield sklearn.utils.shuffle(X_train, y_train)

def train_model(model,train_samples,validation_samples,batch_size):
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    history=model.fit_generator(train_generator, 
                                samples_per_epoch= len(train_samples)*6, 
                                validation_data=validation_generator,
                                nb_val_samples=len(validation_samples)*6, nb_epoch=3)
    return history


@timeit
def train_model(model,train_samples,validation_samples,batch_size):
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    history=model.fit_generator(train_generator, 
                                samples_per_epoch= len(train_samples)*6, 
                                validation_data=validation_generator,
                                nb_val_samples=len(validation_samples)*6, nb_epoch=3)
    return history


@timeit
def main(base,samples=5):
    test_files = glob(base+'test/*.jpg')
    #test_files+= glob("/home/u3920/cervix_type_prediction/test_data/test_stg2/*.jpg")

    train_files = glob(base+'train/*/*.jpg')

    # Additional files
    #train_files+= glob("/data/kaggle_3.27/additional/*/*.jpg")
    shuffle(train_files)
    train_files=train_files[0:samples]
    train_samples, validation_samples = train_test_split(train_files, test_size=0.2)
    old_model_name=None
    model_name="model"
    model=lenet()
    if old_model_name:
        model.load_weights(old_model_name+"_weights.h5")

    print("train_samples",len(train_samples),train_samples[0])
    print("validation_samples",len(validation_samples),validation_samples[0])
    history=train_model(model,train_samples,validation_samples,batch_size=64)
    model.save(model_name+".h5")
    model.save_weights(model_name+"_weights.h5")

    # predictions_df=pd.DataFrame()
    # for x in range(0,len(test_files),400):
    #     start=x
    #     end=min(x+400,len(test_files))
    #     test_image_df,test_imgs_mat = load_test_data(test_files[start:end],X_scaler)
    #     preds=cv.predict(test_imgs_mat)
    #     print("Test set predictions shape",preds.shape)
    #     test_image_df["Type_1"]=preds[:,0]
    #     test_image_df["Type_2"]=preds[:,1]
    #     test_image_df["Type_3"]=preds[:,2]
    #     predictions_df=predictions_df.append(test_image_df,ignore_index=True)

    # func=lambda x: x.split("/")[-1]
    # predictions_df["image_name"]=predictions_df["imagepath"].apply(func)
    # predictions_df[["image_name","Type_1","Type_2","Type_3"]].to_csv(predictions_fi

if __name__=="__main__":
    basepath="/home/carnd/cervix_type_prediction/data/"
    main(basepath,30)
