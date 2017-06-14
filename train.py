#!/usr/bin/env python3
from subprocess import check_output
from glob import glob
import numpy as np
import pandas as pd
#import seaborn as sns

import sklearn
from sklearn.preprocessing import LabelEncoder,Normalizer,StandardScaler
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

from common_functions import *

color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations #18
pix_per_cell = 16# HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
min_size =(480,640)
#y_start_stop = [400, 720] # Min and max in y to search in slide_window()

@timeit
def load_train_data(samples=10,base="/data/kaggle/"):
    print(check_output(["ls", base+"/train"]).decode("utf8"))
        
# Read in cars and notcars
    type_1 = glob(base+'train/Type_1/*.jpg')
    type_2 = glob(base+'train/Type_2/*.jpg')
    type_3 = glob(base+'train/Type_3/*.jpg')
    type_1_features = extract_features(type_1[0:samples], color_space=color_space, 
                                       spatial_size=spatial_size, hist_bins=hist_bins, 
                                       orient=orient, pix_per_cell=pix_per_cell, 
                                       cell_per_block=cell_per_block, 
                                       hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                       hist_feat=hist_feat, hog_feat=hog_feat,
                                       new_size=min_size)
    type_2_features = extract_features(type_2[0:samples], color_space=color_space, 
                                       spatial_size=spatial_size, hist_bins=hist_bins, 
                                       orient=orient, pix_per_cell=pix_per_cell, 
                                       cell_per_block=cell_per_block, 
                                       hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                       hist_feat=hist_feat, hog_feat=hog_feat,
                                       new_size=min_size)

    type_3_features = extract_features(type_3[0:samples], color_space=color_space, 
                                       spatial_size=spatial_size, hist_bins=hist_bins, 
                                       orient=orient, pix_per_cell=pix_per_cell, 
                                       cell_per_block=cell_per_block, 
                                       hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                       hist_feat=hist_feat, hog_feat=hog_feat,
                                       new_size=min_size)    

#    return type_1_features,type_2_features,type_3_features

    X = np.vstack((type_1_features, type_2_features,type_3_features)).astype(np.float64)  

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.zeros(len(type_1_features)), np.ones(len(type_2_features)),2*np.ones(len(type_3_features))))
    
    return scaled_X,y,X_scaler

@timeit
def train_model(base,samples):
    X,y,X_scaler=load_train_data(samples,base)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    clf=svm.SVC(probability=True)
    grid = {
        'C':[1e-5,1e-4,1e-3,1e-2,1e-1, 1, 1e1],
        'gamma': [0.2, 0.45, 0.7,1,10.0],
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

    print("***********************************************************")
    y_test_hat_p = cv.predict_proba(X_test)
    print("Log Loss",sklearn.metrics.log_loss(y_test, y_test_hat_p))
    print("***********************************************************")
    return cv,X_scaler

@timeit
def load_test_data(base,X_scaler):
    test = glob(base+'test/*.jpg')
    test_features = extract_features(test, color_space=color_space, 
                                     spatial_size=spatial_size, hist_bins=hist_bins, 
                                     orient=orient, pix_per_cell=pix_per_cell, 
                                     cell_per_block=cell_per_block, 
                                     hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                     hist_feat=hist_feat, hog_feat=hog_feat,
                                     new_size=min_size)

    X = np.vstack((test_features)).astype(np.float64)  

    # Fit a per-column scaler
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    return scaled_X

@timeit
def main(base,samples=5):
    cv,X_scaler=train_model(base,samples)
    test_images = glob(base+'test/*.jpg')
    test_image_df=pd.DataFrame(pd.DataFrame({'imagepath': test_images}))

    test_imgs_mat = load_test_data(base,X_scaler)

    preds=cv.predict_proba(test_imgs_mat)

    print("Test set predictions shape",preds.shape)

    test_image_df["Type_1"]=preds[:,0]
    test_image_df["Type_2"]=preds[:,1]
    test_image_df["Type_3"]=preds[:,2]
    
    func=lambda x: x.split("/")[-1]
    test_image_df["image_name"]=test_image_df["imagepath"].apply(func)
    
    test_image_df[["image_name","Type_1","Type_2","Type_3"]].to_csv("third_submission.csv",index=False)

if __name__=="__main__":
    basepath="/data/kaggle/"
    print("basepath",basepath)
    main(basepath,10)
