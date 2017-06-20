#!/usr/bin/env python3
from subprocess import check_output
from glob import glob
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from random import shuffle
#import seaborn as sns
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# TODO: Build the Final Test Neural Network in Keras Here
def lenet(input_shape):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    return model

import sklearn
from sklearn.preprocessing import LabelEncoder,Normalizer,StandardScaler
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

from common_functions import *
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 18  # HOG orientations #18
pix_per_cell = 40# HOG pixels per cell. Was 16
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL". Was ALL
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
min_size =(640,480)
vertical_crop=0.15#15% crop on either side
#y_start_stop = [400, 720] # Min and max in y to search in slide_window()
experiment_num="1k"
model_file_name="clf_%s.pkl"%experiment_num
data_file_name="data_%s.npz"%experiment_num
scaler_file_name="standard_scaler_%s.pkl"%experiment_num
predictions_file_name="predictions_%s.csv"%experiment_num

@timeit
def load_train_data(train_files):
    features,labels = extract_features(train_files,
                                       color_space=color_space, 
                                       spatial_size=spatial_size, 
                                       hist_bins=hist_bins, 
                                       orient=orient, 
                                       pix_per_cell=pix_per_cell, 
                                       cell_per_block=cell_per_block, 
                                       hog_channel=hog_channel, 
                                spatial_feat=spatial_feat, 
                                       hist_feat=hist_feat, 
                                       hog_feat=hog_feat,
                                       new_size=min_size,
                                       crop=vertical_crop)
    X = np.vstack((features)).astype(np.float64)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.array(labels)
    print ("labels",labels)
    return scaled_X,y,X_scaler

@timeit
def train_model(X,y,random_state=42):
    input_shape=(min_size[0],min_size[1],3)
    model=lenet(input_shape)
    print("Training data shape",X.shape,"Training label shape",y.shape)
    history = model.fit(X, y, nb_epoch=10, validation_split=0.2)
    return model

@timeit
def load_test_data(test_files,X_scaler=None):
    test_image_df=pd.DataFrame(pd.DataFrame({'imagepath': test_files}))
    test_features,_ = extract_features(test_files, 
                                       color_space=color_space, 
                                       spatial_size=spatial_size, 
                                       hist_bins=hist_bins, 
                                       orient=orient, 
                                       pix_per_cell=pix_per_cell, 
                                       cell_per_block=cell_per_block, 
                                       hog_channel=hog_channel, 
                                       spatial_feat=spatial_feat, 
                                       hist_feat=hist_feat, 
                                       hog_feat=hog_feat,
                                       new_size=min_size,
                                       crop=vertical_crop)

    X = np.vstack((test_features)).astype(np.float64)  
    # Fit a per-column scaler
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    return test_image_df,scaled_X

@timeit
def main(base,samples=5):
    test_files = glob(base+'test/*.jpg')
    test_files+= glob("/home/u3920/cervix_type_prediction/test_data/test_stg2/*.jpg")

    train_files = glob(base+'train/*/*.jpg')
    # Additional files
    train_files+= glob("/data/kaggle_3.27/additional/*/*.jpg")
    shuffle(train_files)

    train_files=train_files[0:samples]

    X,y,X_scaler=load_train_data(train_files)
    label_binarizer = LabelBinarizer()
    y_one_hot = label_binarizer.fit_transform(y)
    cv=train_model(X,y_one_hot)
    
    # try:
    #     joblib.dump(cv,model_file_name)
    # except:
    #     pass

    # predictions_df=pd.DataFrame()
    # for x in range(0,len(test_files),400):
    #     start=x
    #     end=min(x+400,len(test_files))
    #     test_image_df,test_imgs_mat = load_test_data(test_files[start:end],X_scaler)
    #     preds=cv.predict_proba(test_imgs_mat)
    #     print("Test set predictions shape",preds.shape)
    #     test_image_df["Type_1"]=preds[:,0]
    #     test_image_df["Type_2"]=preds[:,1]
    #     test_image_df["Type_3"]=preds[:,2]
    #     predictions_df=predictions_df.append(test_image_df,ignore_index=True)

    # func=lambda x: x.split("/")[-1]
    # predictions_df["image_name"]=predictions_df["imagepath"].apply(func)
    # predictions_df[["image_name","Type_1","Type_2","Type_3"]].to_csv(predictions_file_name,index=False)
    
if __name__=="__main__":
    basepath="/data/kaggle/"
    main(basepath,20)
