#!/usr/bin/env python3
from subprocess import check_output
from glob import glob
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from random import shuffle
#import seaborn as sns

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
spatial_feat = False # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off
min_size =(640,480)
vertical_crop=0.15#15% crop on either side
#y_start_stop = [400, 720] # Min and max in y to search in slide_window()
experiment_num="1j"
model_file_name="clf_%s.pkl"%experiment_num
data_file_name="data_%s.npz"%experiment_num
scaler_file_name="standard_scaler_%s.pkl"%experiment_num
predictions_file_name="predictions_%s.csv"%experiment_num

@timeit
def load_train_data(train_files):
    features = extract_features(train_files,
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
    labels=[get_label(x) for x in train_files]
    # Define the labels vector
    y = np.array(labels)
    print ("labels",labels)
    return scaled_X,y,X_scaler

@timeit
def train_model(X,y,random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)
    print("***********************************************************")

    print("Training data shape",X_train.shape,"Test data shape", X_test.shape, 
          "Training label shape",y_train.shape, "Test label shape",y_test.shape)


    print("Training Label Distribution",pd.Series(y_train).value_counts())
    print("Test Label Distribution",pd.Series(y_test).value_counts())

    clf=svm.SVC(probability=True)
    grid = {
        'C':[1e-5,1e-4,1e-3,1e-2,1e-1, 1, 1e1,1e2,1e3,1e4,1e5],
        'gamma': [1e-5,1e-4,1e-3,1e-2,1e-1, 1, 1e1],
        }
    cv = GridSearchCV(clf, grid, scoring='neg_log_loss', n_jobs=-1, verbose=1)
    cv.fit(X_train, y_train)
    

    for i in range(1, len(cv.cv_results_['params'])+1):
        rank = cv.cv_results_['rank_test_score'][i-1]
        s = cv.cv_results_['mean_test_score'][i-1]
        sd = cv.cv_results_['std_test_score'][i-1]
        params = cv.cv_results_['params'][i-1]
        print("{0}. Mean validation neg log loss: {1:.6f} (std: {2:.6f}) - {3}".format(
                rank,
                s,
                sd,
                params
                ))


    y_test_hat_p = cv.predict_proba(X_test)
    print("Validation log loss",sklearn.metrics.log_loss(y_test, y_test_hat_p))
    print("***********************************************************")
    return cv

@timeit
def load_test_data(test_files,X_scaler=None):
    test_image_df=pd.DataFrame(pd.DataFrame({'imagepath': test_files}))
    test_features = extract_features(test_files, 
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

    try:
        joblib.dump([X,y],data_file_name)
    except:
        pass
    try:
        joblib.dump(X_scaler,scaler_file_name)
    except:
        pass

    cv=train_model(X,y)

    try:
        joblib.dump(cv,model_file_name)
    except:
        pass

    predictions_df=pd.DataFrame()
    for x in range(0,len(test_files),100):
        start=x
        end=min(x+100,len(test_files))
        test_image_df,test_imgs_mat = load_test_data(test_files[start:end],X_scaler)
        preds=cv.predict_proba(test_imgs_mat)
        print("Test set predictions shape",preds.shape)
        test_image_df["Type_1"]=preds[:,0]
        test_image_df["Type_2"]=preds[:,1]
        test_image_df["Type_3"]=preds[:,2]
        predictions_df=predictions_df.append(test_image_df,ignore_index=True)

    func=lambda x: x.split("/")[-1]
    predictions_df["image_name"]=predictions_df["imagepath"].apply(func)
    predictions_df[["image_name","Type_1","Type_2","Type_3"]].to_csv(predictions_file_name,index=False)
    
if __name__=="__main__":
    basepath="/data/kaggle/"
    print("basepath",basepath)
    print("color_space",color_space)
    print("orient",orient)
    print("pix_per_cell",pix_per_cell)
    print("cell_per_block",cell_per_block)
    print("hog_channel",hog_channel)
    print("spatial_feat",spatial_feat)
    print("hist_feat",hist_feat)
    print("hog_feat",hog_feat)
    print("vertical_crop",vertical_crop)
    print("min_size",min_size)
    print("experiment_num",experiment_num)
    print("model_file_name",model_file_name)
    print("data_file_name",data_file_name)
    print("scaler_file_name",scaler_file_name)
    print("predictions_file_name",predictions_file_name)
    main(basepath,1500)
