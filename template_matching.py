#!/usr/bin/env python

import numpy as np
import pandas as pd
from glob import glob
import random
import multiprocessing


# In[13]:


def load_train_files(base,samples=10):
    type_1 = glob(base+'train/Type_1/*.jpg')
    type_2 = glob(base+'train/Type_2/*.jpg')
    type_3 = glob(base+'train/Type_3/*.jpg')
    random.shuffle(type_1)
    random.shuffle(type_2)
    random.shuffle(type_3)
    if samples:
        type_1=type_1[:samples]
        type_2=type_2[:samples]
        type_3=type_3[:samples]
    return type_1,type_2,type_3




# In[21]:


import cv2
import numpy as np

class AKAZEMatcher(object):
    def __init__(self, refImage):
        refImageCopy = refImage.copy()
        self._refImage = cv2.cvtColor(refImageCopy, cv2.COLOR_RGB2GRAY)
        self._algo = cv2.AKAZE_create()
        self._matcher = cv2.FlannBasedMatcher(dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1), dict(checks=100))
        self._kp1, self._des1 = self._algo.detectAndCompute(self._refImage, None)
        
    def match_score(self, repImage):
        repImageCopy = repImage.copy()
        repImageCopy = cv2.cvtColor(repImageCopy, cv2.COLOR_RGB2GRAY)
        kp2, des2 = self._algo.detectAndCompute(repImageCopy, None)
        if ((self._des1 is not None) and (des2 is not None)):
            matches = self._matcher.knnMatch(self._des1, des2, k=2)
            good = 0
            for m_n in matches:
                if len(m_n) != 2:
                    continue
                (m, n) = m_n
                if (m.distance < (0.72 * n.distance)):
                    good = good + 1
            return good
        else:
            return 0
        
    def match_score_debug(self, repImage):
        repImageCopy = repImage.copy()
        repImageCopy = cv2.cvtColor(repImageCopy, cv2.COLOR_RGB2GRAY)
        kp2, des2 = self._algo.detectAndCompute(repImageCopy, None)
        if ((self._des1 is not None) and (des2 is not None)):
            matches = self._matcher.knnMatch(self._des1, des2, k=2)
            good = 0
            matching_features = []
            for m_n in matches:
                if len(m_n) != 2:
                    continue
                (m, n) = m_n
                if (m.distance < (0.72 * n.distance)):
                    good = good + 1
                    matching_features.append([m])
            img_with_features = cv2.drawMatchesKnn(self._refImage, self._kp1, repImageCopy, kp2, matching_features, None ,flags=2)
            return good, img_with_features
        else:
            return 0, None


# In[49]:


def match(args):
    img0=cv2.imread(args[0],-1)
    img1=cv2.imread(args[1],-1)
    type_1=args[2]
    type_2=args[3]
    matcher = AKAZEMatcher(img0)
    match_score, img_with_features = matcher.match_score_debug(img1)
    return type_1,type_2,match_score


def main(base="./"):
    a,b,c=load_train_files(base)
    type_1_pairs=[(a[i],a[j],"Type_1","Type_1") for i in range(len(a)) for j in range(len(a)) if i!=j]
    type_2_pairs=[(b[i],b[j],"Type_2","Type_2") for i in range(len(b)) for j in range(len(b)) if i!=j]
    type_3_pairs=[(c[i],c[j],"Type_3","Type_3") for i in range(len(c)) for j in range(len(c)) if i!=j]
    
    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 2   # arbitrary default

    pool = multiprocessing.Pool(processes=cpus)

    print(match(type_1_pairs[0]))


if __name__=="__main__":
    #main("/data/kaggle/")
    main()
